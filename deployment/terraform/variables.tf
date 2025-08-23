# ============================================
# StockPredictionPro - variables.tf
# Comprehensive input variable definitions with validation, types, and documentation
# ============================================

# ============================================
# Project Configuration Variables
# ============================================

variable "project_name" {
  description = "Name of the project. Used for resource naming and tagging."
  type        = string
  default     = "stockpredpro"
  
  validation {
    condition     = can(regex("^[a-z][a-z0-9-]*[a-z0-9]$", var.project_name))
    error_message = "Project name must start with a letter, contain only lowercase letters, numbers, and hyphens, and end with a letter or number."
  }
}

variable "project_description" {
  description = "Brief description of the project."
  type        = string
  default     = "StockPredictionPro - ML-powered stock prediction and trading signals platform"
}

variable "environment" {
  description = "Environment name (e.g., production, staging, development)."
  type        = string
  
  validation {
    condition     = contains(["production", "staging", "development", "dev", "prod", "stage"], var.environment)
    error_message = "Environment must be one of: production, staging, development, dev, prod, stage."
  }
}

variable "owner" {
  description = "Owner or team responsible for the infrastructure."
  type        = string
  default     = "platform-team"
}

variable "cost_center" {
  description = "Cost center for billing and accounting purposes."
  type        = string
  default     = "engineering"
}

variable "compliance_tags" {
  description = "Tags for compliance and governance requirements."
  type        = map(string)
  default = {
    DataClassification = "financial"
    Compliance         = "SOX,PCI"
    BackupRequired     = "true"
  }
}

variable "application_version" {
  description = "Version of the application being deployed."
  type        = string
  default     = "2.0.0"
  
  validation {
    condition     = can(regex("^\\d+\\.\\d+\\.\\d+(-[a-zA-Z0-9]+)?$", var.application_version))
    error_message = "Application version must follow semantic versioning (e.g., 1.0.0, 2.1.3-beta)."
  }
}

variable "terraform_version" {
  description = "Minimum required Terraform version."
  type        = string
  default     = "~> 1.5"
}

variable "deployment_date" {
  description = "Date of deployment in YYYY-MM-DD format."
  type        = string
  default     = null
  
  validation {
    condition     = var.deployment_date == null || can(regex("^\\d{4}-\\d{2}-\\d{2}$", var.deployment_date))
    error_message = "Deployment date must be in YYYY-MM-DD format."
  }
}

# ============================================
# Cloud Provider Configuration Variables
# ============================================

variable "aws_region" {
  description = "AWS region for resource deployment."
  type        = string
  default     = "us-west-2"
  
  validation {
    condition = can(regex("^[a-z]{2}-[a-z]+-\\d+$", var.aws_region))
    error_message = "AWS region must be a valid region format (e.g., us-west-2, eu-central-1)."
  }
}

variable "aws_availability_zones" {
  description = "List of AWS Availability Zones to use."
  type        = list(string)
  default     = ["us-west-2a", "us-west-2b", "us-west-2c"]
  
  validation {
    condition     = length(var.aws_availability_zones) >= 2
    error_message = "At least 2 availability zones must be specified for high availability."
  }
}

variable "aws_profile" {
  description = "AWS CLI profile to use for authentication."
  type        = string
  default     = null
}

variable "aws_account_id" {
  description = "AWS Account ID for resource validation."
  type        = string
  default     = null
  
  validation {
    condition     = var.aws_account_id == null || can(regex("^\\d{12}$", var.aws_account_id))
    error_message = "AWS Account ID must be exactly 12 digits."
  }
}

# GCP Configuration Variables
variable "gcp_project_id" {
  description = "Google Cloud Project ID."
  type        = string
  default     = null
}

variable "gcp_region" {
  description = "Google Cloud region for resource deployment."
  type        = string
  default     = "us-west2"
}

variable "gcp_zones" {
  description = "List of Google Cloud zones to use."
  type        = list(string)
  default     = ["us-west2-a", "us-west2-b", "us-west2-c"]
}

variable "gcp_credentials_file" {
  description = "Path to Google Cloud service account credentials file."
  type        = string
  default     = null
  sensitive   = true
}

# Azure Configuration Variables
variable "azure_subscription_id" {
  description = "Azure Subscription ID."
  type        = string
  default     = null
}

variable "azure_tenant_id" {
  description = "Azure Tenant ID."
  type        = string
  default     = null
}

variable "azure_location" {
  description = "Azure location for resource deployment."
  type        = string
  default     = "West US 2"
}

variable "azure_resource_group" {
  description = "Azure Resource Group name."
  type        = string
  default     = "rg-stockpredpro"
}

# ============================================
# Network Configuration Variables
# ============================================

variable "vpc_cidr" {
  description = "CIDR block for the VPC."
  type        = string
  default     = "10.0.0.0/16"
  
  validation {
    condition     = can(cidrhost(var.vpc_cidr, 0))
    error_message = "VPC CIDR must be a valid IPv4 CIDR block."
  }
}

variable "vpc_enable_dns_hostnames" {
  description = "Enable DNS hostnames in the VPC."
  type        = bool
  default     = true
}

variable "vpc_enable_dns_support" {
  description = "Enable DNS support in the VPC."
  type        = bool
  default     = true
}

variable "vpc_enable_nat_gateway" {
  description = "Enable NAT Gateway for private subnets."
  type        = bool
  default     = true
}

variable "vpc_single_nat_gateway" {
  description = "Use a single NAT Gateway for all private subnets (cost optimization)."
  type        = bool
  default     = false
}

variable "vpc_enable_vpn_gateway" {
  description = "Enable VPN Gateway for the VPC."
  type        = bool
  default     = false
}

variable "public_subnet_cidrs" {
  description = "List of CIDR blocks for public subnets."
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  
  validation {
    condition     = length(var.public_subnet_cidrs) >= 2
    error_message = "At least 2 public subnets must be specified for high availability."
  }
}

variable "private_subnet_cidrs" {
  description = "List of CIDR blocks for private subnets."
  type        = list(string)
  default     = ["10.0.11.0/24", "10.0.12.0/24", "10.0.13.0/24"]
  
  validation {
    condition     = length(var.private_subnet_cidrs) >= 2
    error_message = "At least 2 private subnets must be specified for high availability."
  }
}

variable "database_subnet_cidrs" {
  description = "List of CIDR blocks for database subnets."
  type        = list(string)
  default     = ["10.0.21.0/24", "10.0.22.0/24", "10.0.23.0/24"]
  
  validation {
    condition     = length(var.database_subnet_cidrs) >= 2
    error_message = "At least 2 database subnets must be specified for RDS subnet group."
  }
}

variable "allowed_cidr_blocks" {
  description = "List of CIDR blocks allowed to access internal resources."
  type        = list(string)
  default     = ["10.0.0.0/16"]
}

variable "office_cidr_blocks" {
  description = "List of office CIDR blocks for administrative access."
  type        = list(string)
  default     = []
}

variable "admin_cidr_blocks" {
  description = "List of administrator CIDR blocks for privileged access."
  type        = list(string)
  default     = []
}

# ============================================
# Kubernetes Cluster Configuration Variables
# ============================================

variable "kubernetes_version" {
  description = "Kubernetes version for the cluster."
  type        = string
  default     = "1.28"
  
  validation {
    condition     = can(regex("^1\\.(2[4-9]|[3-9][0-9])$", var.kubernetes_version))
    error_message = "Kubernetes version must be 1.24 or higher."
  }
}

variable "cluster_name" {
  description = "Name of the Kubernetes cluster."
  type        = string
  default     = null
}

variable "cluster_endpoint_private_access" {
  description = "Enable private API server endpoint access."
  type        = bool
  default     = true
}

variable "cluster_endpoint_public_access" {
  description = "Enable public API server endpoint access."
  type        = bool
  default     = true
}

variable "cluster_endpoint_public_access_cidrs" {
  description = "List of CIDR blocks that can access the public API server endpoint."
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "node_groups" {
  description = "Configuration for EKS node groups."
  type = map(object({
    instance_types = list(string)
    scaling_config = object({
      desired_size = number
      max_size     = number
      min_size     = number
    })
    disk_size     = number
    capacity_type = string
    labels        = map(string)
    taints = list(object({
      key    = string
      value  = string
      effect = string
    }))
  }))
  default = {
    application = {
      instance_types = ["t3.large"]
      scaling_config = {
        desired_size = 3
        max_size     = 10
        min_size     = 2
      }
      disk_size     = 50
      capacity_type = "ON_DEMAND"
      labels = {
        "node-type" = "application"
      }
      taints = []
    }
  }
  
  validation {
    condition = alltrue([
      for ng in var.node_groups : ng.scaling_config.min_size <= ng.scaling_config.desired_size &&
      ng.scaling_config.desired_size <= ng.scaling_config.max_size
    ])
    error_message = "Node group scaling: min_size <= desired_size <= max_size."
  }
}

variable "cluster_addons" {
  description = "Map of cluster addon configurations."
  type = map(object({
    version = string
  }))
  default = {
    "coredns" = {
      version = "v1.10.1-eksbuild.2"
    }
    "kube-proxy" = {
      version = "v1.28.1-eksbuild.1"
    }
    "vpc-cni" = {
      version = "v1.13.4-eksbuild.1"
    }
  }
}

# ============================================
# Database Configuration Variables
# ============================================

variable "database_engine" {
  description = "Database engine type."
  type        = string
  default     = "postgres"
  
  validation {
    condition     = contains(["postgres", "mysql", "mariadb"], var.database_engine)
    error_message = "Database engine must be one of: postgres, mysql, mariadb."
  }
}

variable "database_engine_version" {
  description = "Database engine version."
  type        = string
  default     = "15.3"
}

variable "database_instance_class" {
  description = "RDS instance class."
  type        = string
  default     = "db.r5.large"
  
  validation {
    condition     = can(regex("^db\\.[a-z0-9]+\\.[a-z0-9]+$", var.database_instance_class))
    error_message = "Database instance class must be a valid RDS instance type."
  }
}

variable "database_allocated_storage" {
  description = "Initial allocated storage for the database (GB)."
  type        = number
  default     = 100
  
  validation {
    condition     = var.database_allocated_storage >= 20 && var.database_allocated_storage <= 65536
    error_message = "Database allocated storage must be between 20 and 65536 GB."
  }
}

variable "database_max_allocated_storage" {
  description = "Maximum allocated storage for auto-scaling (GB)."
  type        = number
  default     = 1000
  
  validation {
    condition     = var.database_max_allocated_storage >= var.database_allocated_storage
    error_message = "Maximum allocated storage must be greater than or equal to initial allocated storage."
  }
}

variable "database_storage_encrypted" {
  description = "Enable storage encryption for the database."
  type        = bool
  default     = true
}

variable "database_storage_type" {
  description = "Storage type for the database."
  type        = string
  default     = "gp3"
  
  validation {
    condition     = contains(["gp2", "gp3", "io1", "io2"], var.database_storage_type)
    error_message = "Database storage type must be one of: gp2, gp3, io1, io2."
  }
}

variable "database_iops" {
  description = "IOPS for the database storage (only for io1 and io2)."
  type        = number
  default     = null
}

variable "database_name" {
  description = "Name of the database to create."
  type        = string
  default     = "stockpredpro_prod"
  
  validation {
    condition     = can(regex("^[a-zA-Z][a-zA-Z0-9_]*$", var.database_name))
    error_message = "Database name must start with a letter and contain only letters, numbers, and underscores."
  }
}

variable "database_username" {
  description = "Master username for the database."
  type        = string
  default     = "stockpred_admin"
  
  validation {
    condition     = can(regex("^[a-zA-Z][a-zA-Z0-9_]*$", var.database_username))
    error_message = "Database username must start with a letter and contain only letters, numbers, and underscores."
  }
}

variable "database_backup_retention_period" {
  description = "Number of days to retain database backups."
  type        = number
  default     = 30
  
  validation {
    condition     = var.database_backup_retention_period >= 0 && var.database_backup_retention_period <= 35
    error_message = "Backup retention period must be between 0 and 35 days."
  }
}

variable "database_backup_window" {
  description = "Preferred backup window (UTC)."
  type        = string
  default     = "03:00-04:00"
  
  validation {
    condition     = can(regex("^\\d{2}:\\d{2}-\\d{2}:\\d{2}$", var.database_backup_window))
    error_message = "Backup window must be in HH:MM-HH:MM format."
  }
}

variable "database_maintenance_window" {
  description = "Preferred maintenance window (UTC)."
  type        = string
  default     = "sun:04:00-sun:05:00"
  
  validation {
    condition     = can(regex("^(mon|tue|wed|thu|fri|sat|sun):\\d{2}:\\d{2}-(mon|tue|wed|thu|fri|sat|sun):\\d{2}:\\d{2}$", var.database_maintenance_window))
    error_message = "Maintenance window must be in day:HH:MM-day:HH:MM format."
  }
}

variable "database_deletion_protection" {
  description = "Enable deletion protection for the database."
  type        = bool
  default     = true
}

variable "database_skip_final_snapshot" {
  description = "Skip final snapshot when deleting the database."
  type        = bool
  default     = false
}

variable "database_final_snapshot_identifier" {
  description = "Identifier for the final snapshot when deleting the database."
  type        = string
  default     = null
}

variable "database_read_replica_count" {
  description = "Number of read replicas to create."
  type        = number
  default     = 0
  
  validation {
    condition     = var.database_read_replica_count >= 0 && var.database_read_replica_count <= 5
    error_message = "Read replica count must be between 0 and 5."
  }
}

variable "database_read_replica_instance_class" {
  description = "Instance class for read replicas."
  type        = string
  default     = "db.r5.large"
}

# ============================================
# Cache Configuration Variables (Redis)
# ============================================

variable "redis_node_type" {
  description = "ElastiCache Redis node type."
  type        = string
  default     = "cache.r6g.large"
  
  validation {
    condition     = can(regex("^cache\\.[a-z0-9]+\\.[a-z0-9]+$", var.redis_node_type))
    error_message = "Redis node type must be a valid ElastiCache node type."
  }
}

variable "redis_num_cache_clusters" {
  description = "Number of cache clusters in the replication group."
  type        = number
  default     = 2
  
  validation {
    condition     = var.redis_num_cache_clusters >= 1 && var.redis_num_cache_clusters <= 6
    error_message = "Number of cache clusters must be between 1 and 6."
  }
}

variable "redis_engine_version" {
  description = "Redis engine version."
  type        = string
  default     = "7.0"
}

variable "redis_parameter_group_family" {
  description = "Redis parameter group family."
  type        = string
  default     = "redis7.x"
}

variable "redis_port" {
  description = "Port number for Redis."
  type        = number
  default     = 6379
  
  validation {
    condition     = var.redis_port >= 1024 && var.redis_port <= 65535
    error_message = "Redis port must be between 1024 and 65535."
  }
}

variable "redis_automatic_failover_enabled" {
  description = "Enable automatic failover for Redis replication group."
  type        = bool
  default     = true
}

variable "redis_multi_az_enabled" {
  description = "Enable Multi-AZ for Redis replication group."
  type        = bool
  default     = true
}

variable "redis_at_rest_encryption_enabled" {
  description = "Enable at-rest encryption for Redis."
  type        = bool
  default     = true
}

variable "redis_transit_encryption_enabled" {
  description = "Enable in-transit encryption for Redis."
  type        = bool
  default     = true
}

variable "redis_auth_token_enabled" {
  description = "Enable auth token for Redis."
  type        = bool
  default     = true
}

variable "redis_snapshot_retention_limit" {
  description = "Number of days to retain Redis snapshots."
  type        = number
  default     = 7
  
  validation {
    condition     = var.redis_snapshot_retention_limit >= 0 && var.redis_snapshot_retention_limit <= 35
    error_message = "Snapshot retention limit must be between 0 and 35 days."
  }
}

variable "redis_snapshot_window" {
  description = "Daily time range for Redis snapshots (UTC)."
  type        = string
  default     = "03:00-05:00"
  
  validation {
    condition     = can(regex("^\\d{2}:\\d{2}-\\d{2}:\\d{2}$", var.redis_snapshot_window))
    error_message = "Snapshot window must be in HH:MM-HH:MM format."
  }
}

variable "redis_maintenance_window" {
  description = "Weekly time range for Redis maintenance (UTC)."
  type        = string
  default     = "sun:05:00-sun:07:00"
}

# ============================================
# Storage Configuration Variables
# ============================================

variable "storage_buckets" {
  description = "Configuration for S3 buckets or cloud storage."
  type = map(object({
    bucket_name             = string
    versioning_enabled      = bool
    encryption_enabled      = bool
    lifecycle_enabled       = bool
    public_access_blocked   = bool
  }))
  default = {
    models = {
      bucket_name           = "stockpredpro-models"
      versioning_enabled    = true
      encryption_enabled    = true
      lifecycle_enabled     = true
      public_access_blocked = true
    }
  }
}

variable "efs_encrypted" {
  description = "Enable encryption for EFS file system."
  type        = bool
  default     = true
}

variable "efs_performance_mode" {
  description = "EFS performance mode."
  type        = string
  default     = "generalPurpose"
  
  validation {
    condition     = contains(["generalPurpose", "maxIO"], var.efs_performance_mode)
    error_message = "EFS performance mode must be either generalPurpose or maxIO."
  }
}

variable "efs_throughput_mode" {
  description = "EFS throughput mode."
  type        = string
  default     = "provisioned"
  
  validation {
    condition     = contains(["bursting", "provisioned"], var.efs_throughput_mode)
    error_message = "EFS throughput mode must be either bursting or provisioned."
  }
}

variable "efs_provisioned_throughput_in_mibps" {
  description = "Provisioned throughput for EFS in MiB/s."
  type        = number
  default     = 100
  
  validation {
    condition     = var.efs_provisioned_throughput_in_mibps >= 1
    error_message = "EFS provisioned throughput must be at least 1 MiB/s."
  }
}

# ============================================
# Load Balancer Configuration Variables
# ============================================

variable "load_balancer_type" {
  description = "Type of load balancer."
  type        = string
  default     = "application"
  
  validation {
    condition     = contains(["application", "network", "gateway"], var.load_balancer_type)
    error_message = "Load balancer type must be one of: application, network, gateway."
  }
}

variable "load_balancer_internal" {
  description = "Create an internal load balancer."
  type        = bool
  default     = false
}

variable "load_balancer_enable_deletion_protection" {
  description = "Enable deletion protection for load balancer."
  type        = bool
  default     = true
}

variable "load_balancer_enable_cross_zone_load_balancing" {
  description = "Enable cross-zone load balancing."
  type        = bool
  default     = true
}

variable "load_balancer_idle_timeout" {
  description = "Idle timeout for load balancer connections (seconds)."
  type        = number
  default     = 60
  
  validation {
    condition     = var.load_balancer_idle_timeout >= 1 && var.load_balancer_idle_timeout <= 4000
    error_message = "Load balancer idle timeout must be between 1 and 4000 seconds."
  }
}

variable "ssl_certificate_arn" {
  description = "ARN of the SSL certificate for HTTPS listeners."
  type        = string
  default     = null
}

variable "ssl_policy" {
  description = "SSL security policy for HTTPS listeners."
  type        = string
  default     = "ELBSecurityPolicy-TLS-1-2-2017-01"
}

# ============================================
# DNS Configuration Variables
# ============================================

variable "domain_name" {
  description = "Domain name for the application."
  type        = string
  default     = null
  
  validation {
    condition = var.domain_name == null || can(regex("^[a-zA-Z0-9][a-zA-Z0-9-\\.]*[a-zA-Z0-9]$", var.domain_name))
    error_message = "Domain name must be a valid domain format."
  }
}

variable "create_hosted_zone" {
  description = "Create a new hosted zone for the domain."
  type        = bool
  default     = false
}

variable "hosted_zone_id" {
  description = "Hosted zone ID for DNS records."
  type        = string
  default     = null
}

variable "dns_records" {
  description = "DNS records to create."
  type = map(object({
    name  = string
    type  = string
    alias = bool
  }))
  default = {}
}

# ============================================
# Security Configuration Variables
# ============================================

variable "kms_key_rotation_enabled" {
  description = "Enable automatic key rotation for KMS keys."
  type        = bool
  default     = true
}

variable "kms_key_deletion_window" {
  description = "KMS key deletion window in days."
  type        = number
  default     = 30
  
  validation {
    condition     = var.kms_key_deletion_window >= 7 && var.kms_key_deletion_window <= 30
    error_message = "KMS key deletion window must be between 7 and 30 days."
  }
}

variable "kms_key_alias" {
  description = "Alias for the KMS key."
  type        = string
  default     = "alias/stockpredpro"
}

variable "secrets" {
  description = "Secrets to create in AWS Secrets Manager."
  type = map(object({
    description     = string
    generate_random = bool
    random_length   = number
  }))
  default = {}
}

variable "create_service_accounts" {
  description = "Create Kubernetes service accounts."
  type        = bool
  default     = true
}

variable "service_accounts" {
  description = "Kubernetes service accounts to create."
  type = map(object({
    name        = string
    namespace   = string
    annotations = map(string)
  }))
  default = {}
}

# ============================================
# Monitoring Configuration Variables
# ============================================

variable "enable_cloudwatch_logs" {
  description = "Enable CloudWatch logs."
  type        = bool
  default     = true
}

variable "log_group_retention_days" {
  description = "CloudWatch log group retention in days."
  type        = number
  default     = 30
  
  validation {
    condition = contains([1, 3, 5, 7, 14, 30, 60, 90, 120, 150, 180, 365, 400, 545, 731, 1827, 3653], var.log_group_retention_days)
    error_message = "Log group retention days must be a valid CloudWatch retention value."
  }
}

variable "enable_container_insights" {
  description = "Enable Container Insights for EKS."
  type        = bool
  default     = true
}

variable "enable_prometheus" {
  description = "Enable Prometheus monitoring."
  type        = bool
  default     = true
}

variable "prometheus_namespace" {
  description = "Kubernetes namespace for Prometheus."
  type        = string
  default     = "monitoring"
}

variable "prometheus_storage_size" {
  description = "Storage size for Prometheus."
  type        = string
  default     = "50Gi"
}

variable "prometheus_retention" {
  description = "Data retention period for Prometheus."
  type        = string
  default     = "30d"
}

variable "enable_grafana" {
  description = "Enable Grafana dashboards."
  type        = bool
  default     = true
}

variable "grafana_admin_password" {
  description = "Admin password for Grafana (use secrets manager in production)."
  type        = string
  default     = null
  sensitive   = true
}

variable "grafana_domain" {
  description = "Domain name for Grafana."
  type        = string
  default     = null
}

variable "enable_alertmanager" {
  description = "Enable Alertmanager for alerts."
  type        = bool
  default     = true
}

variable "alert_email_recipients" {
  description = "Email addresses for alert notifications."
  type        = list(string)
  default     = []
}

variable "slack_webhook_url" {
  description = "Slack webhook URL for alert notifications."
  type        = string
  default     = null
  sensitive   = true
}

# ============================================
# Backup and Disaster Recovery Variables
# ============================================

variable "backup_schedule" {
  description = "Cron expression for backup schedule."
  type        = string
  default     = "cron(0 2 * * ? *)"
  
  validation {
    condition     = can(regex("^cron\\([0-9\\s\\*\\?\\-\\/,]+\\)$", var.backup_schedule))
    error_message = "Backup schedule must be a valid cron expression."
  }
}

variable "backup_retention_days" {
  description = "Number of days to retain backups."
  type        = number
  default     = 30
  
  validation {
    condition     = var.backup_retention_days >= 1 && var.backup_retention_days <= 365
    error_message = "Backup retention days must be between 1 and 365."
  }
}

variable "backup_cross_region_enabled" {
  description = "Enable cross-region backup replication."
  type        = bool
  default     = true
}

variable "backup_cross_region_destination" {
  description = "Destination region for cross-region backup replication."
  type        = string
  default     = "us-east-1"
}

variable "dr_enabled" {
  description = "Enable disaster recovery setup."
  type        = bool
  default     = true
}

variable "dr_region" {
  description = "Disaster recovery region."
  type        = string
  default     = "us-east-1"
}

variable "dr_rto_minutes" {
  description = "Recovery Time Objective in minutes."
  type        = number
  default     = 60
  
  validation {
    condition     = var.dr_rto_minutes >= 15 && var.dr_rto_minutes <= 1440
    error_message = "RTO must be between 15 minutes and 24 hours."
  }
}

variable "dr_rpo_minutes" {
  description = "Recovery Point Objective in minutes."
  type        = number
  default     = 15
  
  validation {
    condition     = var.dr_rpo_minutes >= 1 && var.dr_rpo_minutes <= 60
    error_message = "RPO must be between 1 and 60 minutes."
  }
}

# ============================================
# Cost Optimization Variables
# ============================================

variable "enable_cluster_autoscaler" {
  description = "Enable Kubernetes cluster autoscaler."
  type        = bool
  default     = true
}

variable "enable_vertical_pod_autoscaler" {
  description = "Enable Kubernetes vertical pod autoscaler."
  type        = bool
  default     = true
}

variable "enable_karpenter" {
  description = "Enable Karpenter for node provisioning."
  type        = bool
  default     = false
}

variable "use_spot_instances" {
  description = "Use spot instances for cost optimization."
  type        = bool
  default     = true
}

variable "spot_instance_types" {
  description = "List of instance types eligible for spot instances."
  type        = list(string)
  default     = ["t3.large", "t3.xlarge", "m5.large", "m5.xlarge"]
}

variable "spot_max_price" {
  description = "Maximum price for spot instances."
  type        = string
  default     = "0.10"
}

variable "reserved_instance_types" {
  description = "Reserved instance configuration for cost optimization."
  type = map(object({
    count          = number
    term           = string
    payment_option = string
  }))
  default = {}
}

# ============================================
# Feature Flags Variables
# ============================================

variable "enable_waf" {
  description = "Enable AWS Web Application Firewall."
  type        = bool
  default     = true
}

variable "enable_shield_advanced" {
  description = "Enable AWS Shield Advanced."
  type        = bool
  default     = false
}

variable "enable_config_rules" {
  description = "Enable AWS Config rules."
  type        = bool
  default     = true
}

variable "enable_cloudtrail" {
  description = "Enable AWS CloudTrail."
  type        = bool
  default     = true
}

variable "enable_guardduty" {
  description = "Enable AWS GuardDuty."
  type        = bool
  default     = true
}

variable "enable_security_hub" {
  description = "Enable AWS Security Hub."
  type        = bool
  default     = true
}

variable "enable_blue_green_deployment" {
  description = "Enable blue-green deployment strategy."
  type        = bool
  default     = false
}

variable "enable_canary_deployment" {
  description = "Enable canary deployment strategy."
  type        = bool
  default     = true
}

variable "enable_service_mesh" {
  description = "Enable service mesh (Istio/Linkerd)."
  type        = bool
  default     = false
}

variable "enable_api_gateway" {
  description = "Enable API Gateway."
  type        = bool
  default     = true
}

# ============================================
# Environment-Specific Variables
# ============================================

variable "create_dev_environment" {
  description = "Create development environment resources."
  type        = bool
  default     = false
}

variable "dev_environment_config" {
  description = "Configuration for development environment."
  type = object({
    cluster_name               = string
    node_instance_types        = list(string)
    node_desired_capacity      = number
    node_max_capacity          = number
    database_instance_class    = string
    redis_node_type           = string
  })
  default = {
    cluster_name            = "stockpredpro-dev"
    node_instance_types     = ["t3.medium"]
    node_desired_capacity   = 1
    node_max_capacity       = 3
    database_instance_class = "db.t3.micro"
    redis_node_type        = "cache.t3.micro"
  }
}

variable "create_staging_environment" {
  description = "Create staging environment resources."
  type        = bool
  default     = false
}

variable "staging_environment_config" {
  description = "Configuration for staging environment."
  type = object({
    cluster_name               = string
    node_instance_types        = list(string)
    node_desired_capacity      = number
    node_max_capacity          = number
    database_instance_class    = string
    redis_node_type           = string
  })
  default = {
    cluster_name            = "stockpredpro-staging"
    node_instance_types     = ["t3.large"]
    node_desired_capacity   = 2
    node_max_capacity       = 5
    database_instance_class = "db.t3.small"
    redis_node_type        = "cache.t3.small"
  }
}

# ============================================
# External Integrations Variables
# ============================================

variable "external_apis" {
  description = "Configuration for external API integrations."
  type = map(object({
    enabled     = bool
    secret_name = string
  }))
  default = {}
}

variable "sentry_dsn_secret_name" {
  description = "Secret name for Sentry DSN."
  type        = string
  default     = null
}

variable "datadog_api_key_secret_name" {
  description = "Secret name for Datadog API key."
  type        = string
  default     = null
}

# ============================================
# CI/CD Configuration Variables
# ============================================

variable "cicd_provider" {
  description = "CI/CD provider for automated deployments."
  type        = string
  default     = "github_actions"
  
  validation {
    condition     = contains(["github_actions", "gitlab_ci", "jenkins", "azure_devops"], var.cicd_provider)
    error_message = "CI/CD provider must be one of: github_actions, gitlab_ci, jenkins, azure_devops."
  }
}

variable "repository_url" {
  description = "Repository URL for the project."
  type        = string
  default     = null
}

variable "branch_patterns" {
  description = "Branch patterns for different environments."
  type = map(list(string))
  default = {
    production  = ["main", "master"]
    staging     = ["staging", "develop"]
    development = ["dev/*", "feature/*"]
  }
}

variable "auto_deploy_enabled" {
  description = "Enable automatic deployments."
  type        = bool
  default     = true
}

variable "auto_deploy_environments" {
  description = "Environments that support automatic deployment."
  type        = list(string)
  default     = ["development", "staging"]
}

variable "manual_approval_environments" {
  description = "Environments that require manual approval for deployment."
  type        = list(string)
  default     = ["production"]
}

# ============================================
# Compliance and Governance Variables
# ============================================

variable "compliance_frameworks" {
  description = "List of compliance frameworks to adhere to."
  type        = list(string)
  default     = ["SOX", "PCI-DSS", "SOC2"]
}

variable "data_residency_requirements" {
  description = "Data residency requirements by region."
  type        = list(string)
  default     = ["US"]
}

variable "audit_logging_required" {
  description = "Enable comprehensive audit logging."
  type        = bool
  default     = true
}

variable "encryption_at_rest_required" {
  description = "Require encryption at rest for all data stores."
  type        = bool
  default     = true
}

variable "encryption_in_transit_required" {
  description = "Require encryption in transit for all communications."
  type        = bool
  default     = true
}

variable "resource_tagging_required" {
  description = "Require consistent resource tagging."
  type        = bool
  default     = true
}

variable "required_tags" {
  description = "List of required tags for all resources."
  type        = list(string)
  default     = ["Environment", "Owner", "CostCenter", "Project"]
}

variable "allowed_instance_types" {
  description = "List of allowed EC2 instance type patterns."
  type        = list(string)
  default     = ["t3.*", "m5.*", "r5.*", "c5.*"]
}

variable "allowed_regions" {
  description = "List of allowed AWS regions for resource deployment."
  type        = list(string)
  default     = ["us-west-2", "us-east-1"]
}

# ============================================
# Additional Configuration Variables
# ============================================

variable "timezone" {
  description = "Timezone for scheduled operations."
  type        = string
  default     = "UTC"
}

variable "enable_access_logs" {
  description = "Enable access logging for load balancers and services."
  type        = bool
  default     = true
}

variable "enable_flow_logs" {
  description = "Enable VPC flow logs."
  type        = bool
  default     = true
}

variable "log_format" {
  description = "Log format for application logs."
  type        = string
  default     = "json"
  
  validation {
    condition     = contains(["json", "text", "structured"], var.log_format)
    error_message = "Log format must be one of: json, text, structured."
  }
}

variable "enable_x_ray_tracing" {
  description = "Enable AWS X-Ray distributed tracing."
  type        = bool
  default     = true
}

variable "enable_enhanced_monitoring" {
  description = "Enable enhanced monitoring for RDS and ElastiCache."
  type        = bool
  default     = true
}

variable "monitoring_interval" {
  description = "Enhanced monitoring interval in seconds."
  type        = number
  default     = 60
  
  validation {
    condition     = contains([15, 30, 60], var.monitoring_interval)
    error_message = "Monitoring interval must be 15, 30, or 60 seconds."
  }
}

variable "custom_domains" {
  description = "List of custom domains for the application."
  type        = list(string)
  default     = []
}

variable "enable_rate_limiting" {
  description = "Enable rate limiting for APIs."
  type        = bool
  default     = true
}

variable "rate_limit_requests_per_minute" {
  description = "Rate limit requests per minute."
  type        = number
  default     = 1000
  
  validation {
    condition     = var.rate_limit_requests_per_minute >= 1
    error_message = "Rate limit must be at least 1 request per minute."
  }
}

variable "enable_ddos_protection" {
  description = "Enable DDoS protection."
  type        = bool
  default     = true
}
