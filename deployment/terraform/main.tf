# ============================================
# StockPredictionPro - main.tf
# Complete infrastructure deployment with multi-cloud support, high availability, and security
# ============================================

# ============================================
# Terraform Configuration
# ============================================

terraform {
  required_version = var.terraform_version
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.20"
    }
    
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.10"
    }
    
    random = {
      source  = "hashicorp/random"
      version = "~> 3.5"
    }
    
    local = {
      source  = "hashicorp/local"
      version = "~> 2.4"
    }
    
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }
  }
  
  # Remote state configuration (customize for your backend)
  backend "s3" {
    # Configure these in backend config file or CLI
    # bucket         = "your-terraform-state-bucket"
    # key            = "stockpredpro/terraform.tfstate"
    # region         = "us-west-2"
    # dynamodb_table = "terraform-state-lock"
    # encrypt        = true
  }
}

# ============================================
# Provider Configuration
# ============================================

# AWS Provider Configuration
provider "aws" {
  region  = var.aws_region
  profile = var.aws_profile

  default_tags {
    tags = local.common_tags
  }

  # Assume role configuration if needed
  # assume_role {
  #   role_arn = "arn:aws:iam::${var.aws_account_id}:role/TerraformRole"
  # }
}

# Additional AWS provider for cross-region resources
provider "aws" {
  alias   = "dr_region"
  region  = var.dr_region
  profile = var.aws_profile

  default_tags {
    tags = local.common_tags
  }
}

# Kubernetes provider - configured after EKS cluster creation
provider "kubernetes" {
  host                   = module.eks.cluster_endpoint
  cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
  
  exec {
    api_version = "client.authentication.k8s.io/v1beta1"
    command     = "aws"
    args        = ["eks", "get-token", "--cluster-name", module.eks.cluster_name]
  }
}

# Helm provider - for Kubernetes applications
provider "helm" {
  kubernetes {
    host                   = module.eks.cluster_endpoint
    cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
    
    exec {
      api_version = "client.authentication.k8s.io/v1beta1"
      command     = "aws"
      args        = ["eks", "get-token", "--cluster-name", module.eks.cluster_name]
    }
  }
}

# ============================================
# Local Values
# ============================================

locals {
  # Common tags for all resources
  common_tags = merge(
    var.compliance_tags,
    {
      Project             = var.project_name
      Environment         = var.environment
      Owner               = var.owner
      CostCenter          = var.cost_center
      ApplicationVersion  = var.application_version
      TerraformManaged    = "true"
      DeploymentDate      = var.deployment_date != null ? var.deployment_date : formatdate("YYYY-MM-DD", timestamp())
      LastModified        = formatdate("YYYY-MM-DD hh:mm:ss ZZZ", timestamp())
    }
  )

  # Naming conventions
  name_prefix = "${var.project_name}-${var.environment}"
  
  # Cluster name
  cluster_name = var.cluster_name != null ? var.cluster_name : "${local.name_prefix}-cluster"
  
  # Availability zones
  availability_zones = length(var.aws_availability_zones) > 0 ? var.aws_availability_zones : data.aws_availability_zones.available.names
  
  # Database configuration
  database_final_snapshot_identifier = var.database_final_snapshot_identifier != null ? var.database_final_snapshot_identifier : "${local.name_prefix}-final-snapshot-${formatdate("YYYY-MM-DD-hhmm", timestamp())}"
  
  # Storage bucket configurations
  storage_buckets_with_prefix = {
    for k, v in var.storage_buckets : k => merge(v, {
      bucket_name = "${local.name_prefix}-${v.bucket_name}"
    })
  }
}

# ============================================
# Data Sources
# ============================================

# Get current AWS account information
data "aws_caller_identity" "current" {}

# Get current AWS region information
data "aws_region" "current" {}

# Get available availability zones
data "aws_availability_zones" "available" {
  state = "available"
}

# Get latest Amazon Linux 2 AMI
data "aws_ami" "amazon_linux" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["amzn2-ami-hvm-*-x86_64-gp2"]
  }
}

# ============================================
# Random Resources for Passwords
# ============================================

# Generate random password for database
resource "random_password" "database_password" {
  length  = 32
  special = true
}

# Generate random auth token for Redis
resource "random_password" "redis_auth_token" {
  length  = 64
  special = false
}

# Generate random JWT secret key
resource "random_password" "jwt_secret" {
  length  = 64
  special = false
}

# Generate random API secret key
resource "random_password" "api_secret" {
  length  = 32
  special = true
}

# ============================================
# KMS Key for Encryption
# ============================================

resource "aws_kms_key" "main" {
  description             = "KMS key for ${var.project_name} ${var.environment} encryption"
  deletion_window_in_days = var.kms_key_deletion_window
  enable_key_rotation     = var.kms_key_rotation_enabled

  tags = merge(
    local.common_tags,
    {
      Name = "${local.name_prefix}-kms-key"
    }
  )
}

resource "aws_kms_alias" "main" {
  name          = var.kms_key_alias
  target_key_id = aws_kms_key.main.key_id
}

# ============================================
# VPC and Networking
# ============================================

module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "${local.name_prefix}-vpc"
  cidr = var.vpc_cidr

  azs             = local.availability_zones
  public_subnets  = var.public_subnet_cidrs
  private_subnets = var.private_subnet_cidrs
  database_subnets = var.database_subnet_cidrs

  # DNS Configuration
  enable_dns_hostnames = var.vpc_enable_dns_hostnames
  enable_dns_support   = var.vpc_enable_dns_support

  # NAT Gateway Configuration
  enable_nat_gateway     = var.vpc_enable_nat_gateway
  single_nat_gateway     = var.vpc_single_nat_gateway
  one_nat_gateway_per_az = !var.vpc_single_nat_gateway

  # VPN Gateway
  enable_vpn_gateway = var.vpc_enable_vpn_gateway

  # Flow Logs
  enable_flow_log                      = var.enable_flow_logs
  create_flow_log_cloudwatch_iam_role  = var.enable_flow_logs
  create_flow_log_cloudwatch_log_group = var.enable_flow_logs

  # Subnet tags for EKS
  public_subnet_tags = {
    "kubernetes.io/role/elb" = "1"
    "kubernetes.io/cluster/${local.cluster_name}" = "shared"
  }

  private_subnet_tags = {
    "kubernetes.io/role/internal-elb" = "1"
    "kubernetes.io/cluster/${local.cluster_name}" = "shared"
  }

  database_subnet_tags = {
    Type = "database"
  }

  tags = local.common_tags
}

# ============================================
# Security Groups
# ============================================

# Security group for EKS cluster
resource "aws_security_group" "eks_cluster" {
  name_prefix = "${local.name_prefix}-eks-cluster-"
  vpc_id      = module.vpc.vpc_id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(
    local.common_tags,
    {
      Name = "${local.name_prefix}-eks-cluster-sg"
    }
  )

  lifecycle {
    create_before_destroy = true
  }
}

# Security group for RDS
resource "aws_security_group" "rds" {
  name_prefix = "${local.name_prefix}-rds-"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.eks_cluster.id]
  }

  tags = merge(
    local.common_tags,
    {
      Name = "${local.name_prefix}-rds-sg"
    }
  )

  lifecycle {
    create_before_destroy = true
  }
}

# Security group for ElastiCache
resource "aws_security_group" "elasticache" {
  name_prefix = "${local.name_prefix}-elasticache-"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port       = var.redis_port
    to_port         = var.redis_port
    protocol        = "tcp"
    security_groups = [aws_security_group.eks_cluster.id]
  }

  tags = merge(
    local.common_tags,
    {
      Name = "${local.name_prefix}-elasticache-sg"
    }
  )

  lifecycle {
    create_before_destroy = true
  }
}

# ============================================
# AWS Secrets Manager
# ============================================

# Database credentials
resource "aws_secretsmanager_secret" "database_credentials" {
  name                    = "${local.name_prefix}/database/credentials"
  description             = "Database credentials for ${var.project_name} ${var.environment}"
  kms_key_id              = aws_kms_key.main.arn
  recovery_window_in_days = 30

  tags = local.common_tags
}

resource "aws_secretsmanager_secret_version" "database_credentials" {
  secret_id = aws_secretsmanager_secret.database_credentials.id
  secret_string = jsonencode({
    username = var.database_username
    password = random_password.database_password.result
    engine   = var.database_engine
    host     = module.rds.db_instance_endpoint
    port     = module.rds.db_instance_port
    dbname   = var.database_name
  })
}

# Redis auth token
resource "aws_secretsmanager_secret" "redis_auth_token" {
  name                    = "${local.name_prefix}/redis/auth-token"
  description             = "Redis authentication token for ${var.project_name} ${var.environment}"
  kms_key_id              = aws_kms_key.main.arn
  recovery_window_in_days = 30

  tags = local.common_tags
}

resource "aws_secretsmanager_secret_version" "redis_auth_token" {
  secret_id = aws_secretsmanager_secret.redis_auth_token.id
  secret_string = jsonencode({
    auth_token = random_password.redis_auth_token.result
    host       = module.elasticache.primary_endpoint_address
    port       = var.redis_port
  })
}

# API secrets
resource "aws_secretsmanager_secret" "api_secrets" {
  name                    = "${local.name_prefix}/api/secrets"
  description             = "API secrets for ${var.project_name} ${var.environment}"
  kms_key_id              = aws_kms_key.main.arn
  recovery_window_in_days = 30

  tags = local.common_tags
}

resource "aws_secretsmanager_secret_version" "api_secrets" {
  secret_id = aws_secretsmanager_secret.api_secrets.id
  secret_string = jsonencode({
    jwt_secret_key = random_password.jwt_secret.result
    api_secret_key = random_password.api_secret.result
  })
}

# External API keys
resource "aws_secretsmanager_secret" "external_api_keys" {
  for_each = var.external_apis

  name                    = each.value.secret_name
  description             = "External API key for ${each.key}"
  kms_key_id              = aws_kms_key.main.arn
  recovery_window_in_days = 30

  tags = local.common_tags
}

# ============================================
# S3 Buckets
# ============================================

module "s3_buckets" {
  source = "terraform-aws-modules/s3-bucket/aws"
  version = "~> 3.0"

  for_each = local.storage_buckets_with_prefix

  bucket = each.value.bucket_name

  # Versioning
  versioning = {
    enabled = each.value.versioning_enabled
  }

  # Server-side encryption
  server_side_encryption_configuration = {
    rule = {
      apply_server_side_encryption_by_default = {
        kms_master_key_id = aws_kms_key.main.arn
        sse_algorithm     = "aws:kms"
      }
    }
  }

  # Public access block
  block_public_acls       = each.value.public_access_blocked
  block_public_policy     = each.value.public_access_blocked
  ignore_public_acls      = each.value.public_access_blocked
  restrict_public_buckets = each.value.public_access_blocked

  # Lifecycle configuration
  lifecycle_rule = each.value.lifecycle_enabled ? [
    {
      id     = "lifecycle"
      status = "Enabled"

      transition = [
        {
          days          = 30
          storage_class = "STANDARD_IA"
        },
        {
          days          = 90
          storage_class = "GLACIER"
        },
        {
          days          = 365
          storage_class = "DEEP_ARCHIVE"
        }
      ]

      expiration = {
        days = 2555  # 7 years
      }

      noncurrent_version_expiration = {
        days = 90
      }
    }
  ] : []

  tags = merge(
    local.common_tags,
    {
      Name = each.value.bucket_name
      Type = each.key
    }
  )
}

# ============================================
# RDS PostgreSQL Database
# ============================================

module "rds" {
  source = "terraform-aws-modules/rds/aws"
  version = "~> 6.0"

  identifier = "${local.name_prefix}-db"

  # Database configuration
  engine         = var.database_engine
  engine_version = var.database_engine_version
  instance_class = var.database_instance_class

  # Storage configuration
  allocated_storage     = var.database_allocated_storage
  max_allocated_storage = var.database_max_allocated_storage
  storage_encrypted     = var.database_storage_encrypted
  storage_type          = var.database_storage_type
  iops                  = var.database_storage_type == "io1" || var.database_storage_type == "io2" ? var.database_iops : null
  kms_key_id            = aws_kms_key.main.arn

  # Database credentials
  db_name  = var.database_name
  username = var.database_username
  password = random_password.database_password.result

  # Network configuration
  db_subnet_group_name   = module.vpc.database_subnet_group_name
  vpc_security_group_ids = [aws_security_group.rds.id]

  # Backup configuration
  backup_retention_period = var.database_backup_retention_period
  backup_window          = var.database_backup_window
  maintenance_window     = var.database_maintenance_window

  # Monitoring
  monitoring_interval = var.enable_enhanced_monitoring ? var.monitoring_interval : 0
  monitoring_role_arn = var.enable_enhanced_monitoring ? aws_iam_role.rds_enhanced_monitoring[0].arn : null

  # Security
  deletion_protection       = var.database_deletion_protection
  skip_final_snapshot      = var.database_skip_final_snapshot
  final_snapshot_identifier = var.database_skip_final_snapshot ? null : local.database_final_snapshot_identifier

  # Performance Insights
  performance_insights_enabled = true
  performance_insights_kms_key_id = aws_kms_key.main.arn

  tags = merge(
    local.common_tags,
    {
      Name = "${local.name_prefix}-database"
    }
  )
}

# RDS Enhanced Monitoring IAM Role
resource "aws_iam_role" "rds_enhanced_monitoring" {
  count = var.enable_enhanced_monitoring ? 1 : 0

  name = "${local.name_prefix}-rds-enhanced-monitoring"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "monitoring.rds.amazonaws.com"
        }
      }
    ]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "rds_enhanced_monitoring" {
  count = var.enable_enhanced_monitoring ? 1 : 0

  role       = aws_iam_role.rds_enhanced_monitoring[0].name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonRDSEnhancedMonitoringRole"
}

# Read Replicas
module "rds_replica" {
  source = "terraform-aws-modules/rds/aws"
  version = "~> 6.0"

  count = var.database_read_replica_count

  identifier = "${local.name_prefix}-db-replica-${count.index + 1}"

  # Replica configuration
  replicate_source_db = module.rds.db_instance_identifier
  instance_class      = var.database_read_replica_instance_class

  # Storage
  storage_encrypted = var.database_storage_encrypted
  kms_key_id        = aws_kms_key.main.arn

  # Network
  vpc_security_group_ids = [aws_security_group.rds.id]

  # Monitoring
  monitoring_interval = var.enable_enhanced_monitoring ? var.monitoring_interval : 0
  monitoring_role_arn = var.enable_enhanced_monitoring ? aws_iam_role.rds_enhanced_monitoring[0].arn : null

  # Performance Insights
  performance_insights_enabled = true
  performance_insights_kms_key_id = aws_kms_key.main.arn

  tags = merge(
    local.common_tags,
    {
      Name = "${local.name_prefix}-database-replica-${count.index + 1}"
      Type = "read-replica"
    }
  )

  depends_on = [module.rds]
}

# ============================================
# ElastiCache Redis
# ============================================

module "elasticache" {
  source = "terraform-aws-modules/elasticache/aws"
  version = "~> 1.0"

  # Replication group configuration
  replication_group_id         = "${local.name_prefix}-redis"
  description                  = "Redis cluster for ${var.project_name} ${var.environment}"
  
  # Engine configuration
  engine_version      = var.redis_engine_version
  node_type          = var.redis_node_type
  parameter_group_name = aws_elasticache_parameter_group.redis.name
  port               = var.redis_port

  # Cluster configuration
  num_cache_clusters         = var.redis_num_cache_clusters
  automatic_failover_enabled = var.redis_automatic_failover_enabled
  multi_az_enabled          = var.redis_multi_az_enabled

  # Security
  at_rest_encryption_enabled = var.redis_at_rest_encryption_enabled
  transit_encryption_enabled = var.redis_transit_encryption_enabled
  auth_token                = var.redis_auth_token_enabled ? random_password.redis_auth_token.result : null
  kms_key_id               = aws_kms_key.main.arn

  # Network configuration
  subnet_group_name = aws_elasticache_subnet_group.redis.name
  security_group_ids = [aws_security_group.elasticache.id]

  # Backup configuration
  snapshot_retention_limit = var.redis_snapshot_retention_limit
  snapshot_window         = var.redis_snapshot_window
  maintenance_window      = var.redis_maintenance_window

  tags = merge(
    local.common_tags,
    {
      Name = "${local.name_prefix}-redis"
    }
  )
}

# ElastiCache subnet group
resource "aws_elasticache_subnet_group" "redis" {
  name       = "${local.name_prefix}-redis-subnet-group"
  subnet_ids = module.vpc.private_subnets

  tags = local.common_tags
}

# ElastiCache parameter group
resource "aws_elasticache_parameter_group" "redis" {
  family = var.redis_parameter_group_family
  name   = "${local.name_prefix}-redis-params"

  parameter {
    name  = "maxmemory-policy"
    value = "allkeys-lru"
  }

  tags = local.common_tags
}

# ============================================
# EKS Cluster
# ============================================

module "eks" {
  source = "terraform-aws-modules/eks/aws"
  version = "~> 19.0"

  cluster_name    = local.cluster_name
  cluster_version = var.kubernetes_version

  # Networking
  vpc_id                   = module.vpc.vpc_id
  subnet_ids              = module.vpc.private_subnets
  control_plane_subnet_ids = module.vpc.public_subnets

  # Cluster endpoint configuration
  cluster_endpoint_private_access = var.cluster_endpoint_private_access
  cluster_endpoint_public_access  = var.cluster_endpoint_public_access
  cluster_endpoint_public_access_cidrs = var.cluster_endpoint_public_access_cidrs

  # Security
  cluster_additional_security_group_ids = [aws_security_group.eks_cluster.id]
  cluster_encryption_config = [
    {
      provider_key_arn = aws_kms_key.main.arn
      resources        = ["secrets"]
    }
  ]

  # Logging
  cluster_enabled_log_types = ["api", "audit", "authenticator", "controllerManager", "scheduler"]

  # Add-ons
  cluster_addons = {
    for addon_name, addon_config in var.cluster_addons : addon_name => {
      most_recent = addon_config.version == null
      addon_version = addon_config.version
    }
  }

  # Node groups
  eks_managed_node_groups = {
    for name, config in var.node_groups : name => {
      # Instance configuration
      instance_types = config.instance_types
      capacity_type  = config.capacity_type

      # Scaling configuration
      min_size     = config.scaling_config.min_size
      max_size     = config.scaling_config.max_size
      desired_size = config.scaling_config.desired_size

      # Disk configuration
      disk_size = config.disk_size

      # Labels and taints
      labels = config.labels
      taints = config.taints

      # AMI configuration
      ami_type = "AL2_x86_64"

      # Update configuration
      update_config = {
        max_unavailable_percentage = 25
      }

      tags = merge(
        local.common_tags,
        {
          Name = "${local.name_prefix}-${name}-nodes"
        }
      )
    }
  }

  # Enable IRSA
  enable_irsa = true

  tags = local.common_tags
}

# ============================================
# Application Load Balancer
# ============================================

module "alb" {
  source = "terraform-aws-modules/alb/aws"
  version = "~> 8.0"

  name = "${local.name_prefix}-alb"

  load_balancer_type = var.load_balancer_type
  internal          = var.load_balancer_internal

  vpc_id  = module.vpc.vpc_id
  subnets = module.vpc.public_subnets

  # Security groups
  security_groups = [aws_security_group.alb.id]

  # Attributes
  enable_deletion_protection     = var.load_balancer_enable_deletion_protection
  enable_cross_zone_load_balancing = var.load_balancer_enable_cross_zone_load_balancing
  idle_timeout                   = var.load_balancer_idle_timeout

  # Access logs
  access_logs = var.enable_access_logs ? {
    bucket  = module.s3_buckets["logs"].s3_bucket_id
    prefix  = "alb-access-logs"
    enabled = true
  } : {}

  # Listeners
  http_tcp_listeners = [
    {
      port               = 80
      protocol           = "HTTP"
      action_type        = "redirect"
      redirect = {
        port        = "443"
        protocol    = "HTTPS"
        status_code = "HTTP_301"
      }
    }
  ]

  https_listeners = var.ssl_certificate_arn != null ? [
    {
      port               = 443
      protocol           = "HTTPS"
      certificate_arn    = var.ssl_certificate_arn
      ssl_policy         = var.ssl_policy
      action_type        = "forward"
      target_group_index = 0
    }
  ] : []

  # Target groups
  target_groups = [
    {
      name_prefix      = "${substr(local.name_prefix, 0, 6)}-"
      backend_protocol = "HTTP"
      backend_port     = var.target_group_port
      target_type      = "ip"
      
      health_check = {
        enabled             = var.target_group_health_check_enabled
        healthy_threshold   = var.target_group_healthy_threshold
        interval            = var.target_group_health_check_interval
        matcher             = "200"
        path                = var.target_group_health_check_path
        port                = "traffic-port"
        protocol            = var.target_group_protocol
        timeout             = var.target_group_health_check_timeout
        unhealthy_threshold = var.target_group_unhealthy_threshold
      }
    }
  ]

  tags = local.common_tags
}

# ALB Security Group
resource "aws_security_group" "alb" {
  name_prefix = "${local.name_prefix}-alb-"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(
    local.common_tags,
    {
      Name = "${local.name_prefix}-alb-sg"
    }
  )

  lifecycle {
    create_before_destroy = true
  }
}

# ============================================
# Route53 DNS
# ============================================

# Hosted Zone (if creating new)
resource "aws_route53_zone" "main" {
  count = var.create_hosted_zone && var.domain_name != null ? 1 : 0

  name = var.domain_name

  tags = merge(
    local.common_tags,
    {
      Name = var.domain_name
    }
  )
}

# DNS Records
resource "aws_route53_record" "dns_records" {
  for_each = var.dns_records

  zone_id = var.hosted_zone_id != null ? var.hosted_zone_id : (var.create_hosted_zone ? aws_route53_zone.main[0].zone_id : null)
  name    = each.value.name
  type    = each.value.type

  dynamic "alias" {
    for_each = each.value.alias ?  : []
    content {
      name                   = module.alb.lb_dns_name
      zone_id               = module.alb.lb_zone_id
      evaluate_target_health = true
    }
  }
}

# ============================================
# CloudWatch Log Groups
# ============================================

resource "aws_cloudwatch_log_group" "application" {
  count = var.enable_cloudwatch_logs ? 1 : 0

  name              = "/aws/eks/${local.cluster_name}/application"
  retention_in_days = var.log_group_retention_days
  kms_key_id        = aws_kms_key.main.arn

  tags = local.common_tags
}

# ============================================
# IAM Roles for Service Accounts (IRSA)
# ============================================

# EBS CSI Driver IAM role
module "ebs_csi_irsa_role" {
  source = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.0"

  role_name             = "${local.name_prefix}-ebs-csi-driver"
  attach_ebs_csi_policy = true

  oidc_providers = {
    ex = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["kube-system:ebs-csi-controller-sa"]
    }
  }

  tags = local.common_tags
}

# Load Balancer Controller IAM role
module "load_balancer_controller_irsa_role" {
  source = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.0"

  role_name                              = "${local.name_prefix}-load-balancer-controller"
  attach_load_balancer_controller_policy = true

  oidc_providers = {
    ex = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["kube-system:aws-load-balancer-controller"]
    }
  }

  tags = local.common_tags
}

# Application IAM role
module "application_irsa_role" {
  source = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.0"

  role_name = "${local.name_prefix}-application"

  role_policy_arns = {
    policy = aws_iam_policy.application.arn
  }

  oidc_providers = {
    ex = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["stockpredpro:stockpredpro-service-account"]
    }
  }

  tags = local.common_tags
}

# Application IAM policy
resource "aws_iam_policy" "application" {
  name_prefix = "${local.name_prefix}-application"
  description = "IAM policy for StockPredictionPro application"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject"
        ]
        Resource = [
          for bucket in module.s3_buckets : "${bucket.s3_bucket_arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "s3:ListBucket"
        ]
        Resource = [
          for bucket in module.s3_buckets : bucket.s3_bucket_arn
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue"
        ]
        Resource = [
          aws_secretsmanager_secret.database_credentials.arn,
          aws_secretsmanager_secret.redis_auth_token.arn,
          aws_secretsmanager_secret.api_secrets.arn
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "kms:Decrypt",
          "kms:DescribeKey"
        ]
        Resource = [aws_kms_key.main.arn]
      }
    ]
  })

  tags = local.common_tags
}

# ============================================
# Monitoring and Logging
# ============================================

# Container Insights
resource "aws_eks_addon" "container_insights" {
  count = var.enable_container_insights ? 1 : 0

  cluster_name = module.eks.cluster_name
  addon_name   = "amazon-cloudwatch-observability"

  tags = local.common_tags

  depends_on = [module.eks]
}

# ============================================
# Conditional Environment Resources
# ============================================

# Development Environment
module "dev_environment" {
  source = "./modules/environment"
  count  = var.create_dev_environment ? 1 : 0

  project_name    = var.project_name
  environment     = "development"
  vpc_cidr        = "10.1.0.0/16"
  config          = var.dev_environment_config
  common_tags     = local.common_tags

  # Pass necessary variables
  aws_region = var.aws_region
  kms_key_arn = aws_kms_key.main.arn
}

# Staging Environment
module "staging_environment" {
  source = "./modules/environment"
  count  = var.create_staging_environment ? 1 : 0

  project_name    = var.project_name
  environment     = "staging"
  vpc_cidr        = "10.2.0.0/16"
  config          = var.staging_environment_config
  common_tags     = local.common_tags

  # Pass necessary variables
  aws_region = var.aws_region
  kms_key_arn = aws_kms_key.main.arn
}

# ============================================
# Security and Compliance
# ============================================

# AWS Config (if enabled)
module "aws_config" {
  source = "terraform-aws-modules/config/aws"
  version = "~> 1.0"

  count = var.enable_config_rules ? 1 : 0

  configuration_recorder_name = "${local.name_prefix}-config-recorder"
  deliverys3_bucket_name     = "${local.name_prefix}-config-bucket"

  tags = local.common_tags
}

# GuardDuty (if enabled)
resource "aws_guardduty_detector" "main" {
  count = var.enable_guardduty ? 1 : 0

  enable = true

  datasources {
    s3_logs {
      enable = true
    }
    kubernetes {
      audit_logs {
        enable = true
      }
    }
    malware_protection {
      scan_ec2_instance_with_findings {
        ebs_volumes {
          enable = true
        }
      }
    }
  }

  tags = local.common_tags
}

# ============================================
# Backup Configuration
# ============================================

# AWS Backup Vault
resource "aws_backup_vault" "main" {
  name        = "${local.name_prefix}-backup-vault"
  kms_key_arn = aws_kms_key.main.arn

  tags = local.common_tags
}

# Backup Plan
resource "aws_backup_plan" "main" {
  name = "${local.name_prefix}-backup-plan"

  rule {
    rule_name         = "daily_backups"
    target_vault_name = aws_backup_vault.main.name
    schedule          = var.backup_schedule

    lifecycle {
      cold_storage_after = 30
      delete_after       = var.backup_retention_days
    }

    recovery_point_tags = local.common_tags
  }

  tags = local.common_tags
}

# Backup IAM Role
resource "aws_iam_role" "backup" {
  name = "${local.name_prefix}-backup-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "backup.amazonaws.com"
        }
      }
    ]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "backup" {
  role       = aws_iam_role.backup.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSBackupServiceRolePolicyForBackup"
}

# Backup Selection
resource "aws_backup_selection" "main" {
  iam_role_arn = aws_iam_role.backup.arn
  name         = "${local.name_prefix}-backup-selection"
  plan_id      = aws_backup_plan.main.id

  resources = [
    module.rds.db_instance_arn
  ]

  condition {
    string_equals {
      key   = "Environment"
      value = var.environment
    }
  }
}
