# ============================================
# StockPredictionPro - outputs.tf
# Comprehensive infrastructure outputs for integration, monitoring, and operational visibility
# ============================================

# ============================================
# Project Information Outputs
# ============================================

output "project_name" {
  description = "Name of the project"
  value       = var.project_name
}

output "environment" {
  description = "Environment name"
  value       = var.environment
}

output "region" {
  description = "AWS region where infrastructure is deployed"
  value       = data.aws_region.current.name
}

output "availability_zones" {
  description = "List of availability zones used"
  value       = local.availability_zones
}

output "deployment_timestamp" {
  description = "Timestamp when the infrastructure was deployed"
  value       = timestamp()
}

# ============================================
# VPC and Networking Outputs
# ============================================

output "vpc_id" {
  description = "ID of the VPC"
  value       = module.vpc.vpc_id
}

output "vpc_cidr_block" {
  description = "CIDR block of the VPC"
  value       = module.vpc.vpc_cidr_block
}

output "public_subnet_ids" {
  description = "List of IDs of the public subnets"
  value       = module.vpc.public_subnets
}

output "private_subnet_ids" {
  description = "List of IDs of the private subnets"
  value       = module.vpc.private_subnets
}

output "database_subnet_ids" {
  description = "List of IDs of the database subnets"
  value       = module.vpc.database_subnets
}

output "database_subnet_group_name" {
  description = "Name of the database subnet group"
  value       = module.vpc.database_subnet_group_name
}

output "nat_gateway_ids" {
  description = "List of IDs of the NAT Gateways"
  value       = module.vpc.natgw_ids
}

output "internet_gateway_id" {
  description = "ID of the Internet Gateway"
  value       = module.vpc.igw_id
}

# ============================================
# Security Group Outputs
# ============================================

output "eks_cluster_security_group_id" {
  description = "Security group ID for EKS cluster"
  value       = aws_security_group.eks_cluster.id
}

output "rds_security_group_id" {
  description = "Security group ID for RDS database"
  value       = aws_security_group.rds.id
}

output "elasticache_security_group_id" {
  description = "Security group ID for ElastiCache"
  value       = aws_security_group.elasticache.id
}

output "alb_security_group_id" {
  description = "Security group ID for Application Load Balancer"
  value       = aws_security_group.alb.id
}

# ============================================
# EKS Cluster Outputs
# ============================================

output "cluster_name" {
  description = "Name of the EKS cluster"
  value       = module.eks.cluster_name
}

output "cluster_arn" {
  description = "ARN of the EKS cluster"
  value       = module.eks.cluster_arn
}

output "cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = module.eks.cluster_endpoint
  sensitive   = true
}

output "cluster_version" {
  description = "The Kubernetes version for the EKS cluster"
  value       = module.eks.cluster_version
}

output "cluster_platform_version" {
  description = "Platform version for the EKS cluster"
  value       = module.eks.cluster_platform_version
}

output "cluster_certificate_authority_data" {
  description = "Base64 encoded certificate data required to communicate with the cluster"
  value       = module.eks.cluster_certificate_authority_data
  sensitive   = true
}

output "cluster_oidc_issuer_url" {
  description = "The URL on the EKS cluster OIDC Issuer"
  value       = module.eks.cluster_oidc_issuer_url
}

output "oidc_provider_arn" {
  description = "The ARN of the OIDC Provider if enabled"
  value       = module.eks.oidc_provider_arn
}

output "node_groups" {
  description = "EKS node groups configuration"
  value = {
    for name, config in var.node_groups : name => {
      node_group_arn    = try(module.eks.eks_managed_node_groups[name].node_group_arn, null)
      node_group_status = try(module.eks.eks_managed_node_groups[name].node_group_status, null)
      capacity_type     = config.capacity_type
      instance_types    = config.instance_types
      scaling_config    = config.scaling_config
    }
  }
}

# ============================================
# Database Outputs
# ============================================

output "database_instance_id" {
  description = "RDS instance ID"
  value       = module.rds.db_instance_identifier
}

output "database_instance_arn" {
  description = "RDS instance ARN"
  value       = module.rds.db_instance_arn
}

output "database_endpoint" {
  description = "RDS instance endpoint"
  value       = module.rds.db_instance_endpoint
  sensitive   = true
}

output "database_port" {
  description = "RDS instance port"
  value       = module.rds.db_instance_port
}

output "database_name" {
  description = "Database name"
  value       = var.database_name
}

output "database_username" {
  description = "Database master username"
  value       = var.database_username
  sensitive   = true
}

output "database_engine" {
  description = "Database engine"
  value       = var.database_engine
}

output "database_engine_version" {
  description = "Database engine version"
  value       = var.database_engine_version
}

output "database_read_replica_endpoints" {
  description = "Read replica endpoints"
  value = var.database_read_replica_count > 0 ? [
    for replica in module.rds_replica : replica.db_instance_endpoint
  ] : []
  sensitive = true
}

output "database_backup_retention_period" {
  description = "Database backup retention period"
  value       = var.database_backup_retention_period
}

# ============================================
# Cache (Redis) Outputs
# ============================================

output "redis_cluster_id" {
  description = "ElastiCache replication group ID"
  value       = module.elasticache.replication_group_id
}

output "redis_cluster_arn" {
  description = "ElastiCache replication group ARN"
  value       = module.elasticache.replication_group_arn
}

output "redis_primary_endpoint" {
  description = "Redis primary endpoint"
  value       = module.elasticache.primary_endpoint_address
  sensitive   = true
}

output "redis_reader_endpoint" {
  description = "Redis reader endpoint"
  value       = module.elasticache.reader_endpoint_address
  sensitive   = true
}

output "redis_port" {
  description = "Redis port"
  value       = var.redis_port
}

output "redis_node_type" {
  description = "Redis node type"
  value       = var.redis_node_type
}

output "redis_num_cache_clusters" {
  description = "Number of cache clusters"
  value       = var.redis_num_cache_clusters
}

output "redis_auth_token_enabled" {
  description = "Whether auth token is enabled for Redis"
  value       = var.redis_auth_token_enabled
}

# ============================================
# Storage Outputs
# ============================================

output "s3_buckets" {
  description = "S3 bucket information"
  value = {
    for name, bucket in module.s3_buckets : name => {
      id     = bucket.s3_bucket_id
      arn    = bucket.s3_bucket_arn
      domain = bucket.s3_bucket_bucket_domain_name
      region = bucket.s3_bucket_region
    }
  }
}

output "s3_bucket_names" {
  description = "Names of created S3 buckets"
  value = {
    for name, config in local.storage_buckets_with_prefix : name => config.bucket_name
  }
}

# ============================================
# Load Balancer Outputs
# ============================================

output "load_balancer_arn" {
  description = "ARN of the load balancer"
  value       = module.alb.lb_arn
}

output "load_balancer_dns_name" {
  description = "DNS name of the load balancer"
  value       = module.alb.lb_dns_name
}

output "load_balancer_zone_id" {
  description = "Zone ID of the load balancer"
  value       = module.alb.lb_zone_id
}

output "load_balancer_hosted_zone_id" {
  description = "Hosted zone ID of the load balancer"
  value       = module.alb.lb_zone_id
}

output "target_group_arns" {
  description = "ARNs of the target groups"
  value       = module.alb.target_group_arns
}

# ============================================
# DNS Outputs
# ============================================

output "domain_name" {
  description = "Primary domain name"
  value       = var.domain_name
}

output "hosted_zone_id" {
  description = "Route53 hosted zone ID"
  value       = var.hosted_zone_id != null ? var.hosted_zone_id : (var.create_hosted_zone ? aws_route53_zone.main[0].zone_id : null)
}

output "hosted_zone_name_servers" {
  description = "Name servers for the hosted zone"
  value       = var.create_hosted_zone && var.domain_name != null ? aws_route53_zone.main[0].name_servers : null
}

output "dns_records" {
  description = "Created DNS records"
  value = {
    for name, record in aws_route53_record.dns_records : name => {
      name  = record.name
      type  = record.type
      fqdn  = record.fqdn
    }
  }
}

# ============================================
# Security and Compliance Outputs
# ============================================

output "kms_key_id" {
  description = "KMS key ID"
  value       = aws_kms_key.main.id
}

output "kms_key_arn" {
  description = "KMS key ARN"
  value       = aws_kms_key.main.arn
}

output "kms_key_alias" {
  description = "KMS key alias"
  value       = aws_kms_alias.main.name
}

output "secrets_manager_arns" {
  description = "ARNs of Secrets Manager secrets"
  value = {
    database_credentials = aws_secretsmanager_secret.database_credentials.arn
    redis_auth_token    = aws_secretsmanager_secret.redis_auth_token.arn
    api_secrets         = aws_secretsmanager_secret.api_secrets.arn
  }
}

output "backup_vault_arn" {
  description = "ARN of the backup vault"
  value       = aws_backup_vault.main.arn
}

output "backup_plan_arn" {
  description = "ARN of the backup plan"
  value       = aws_backup_plan.main.arn
}

# ============================================
# IAM Outputs
# ============================================

output "iam_roles" {
  description = "IAM roles created for the infrastructure"
  value = {
    ebs_csi_driver         = module.ebs_csi_irsa_role.iam_role_arn
    load_balancer_controller = module.load_balancer_controller_irsa_role.iam_role_arn
    application            = module.application_irsa_role.iam_role_arn
    backup                 = aws_iam_role.backup.arn
    rds_enhanced_monitoring = var.enable_enhanced_monitoring ? aws_iam_role.rds_enhanced_monitoring[0].arn : null
  }
}

output "application_iam_policy_arn" {
  description = "ARN of the application IAM policy"
  value       = aws_iam_policy.application.arn
}

# ============================================
# Monitoring Outputs
# ============================================

output "cloudwatch_log_groups" {
  description = "CloudWatch log group information"
  value = var.enable_cloudwatch_logs ? {
    application = {
      name = aws_cloudwatch_log_group.application[0].name
      arn  = aws_cloudwatch_log_group.application.arn
    }
  } : {}
}

output "monitoring_enabled" {
  description = "Monitoring features enabled"
  value = {
    cloudwatch_logs     = var.enable_cloudwatch_logs
    container_insights  = var.enable_container_insights
    prometheus         = var.enable_prometheus
    grafana           = var.enable_grafana
    alertmanager      = var.enable_alertmanager
    enhanced_monitoring = var.enable_enhanced_monitoring
    x_ray_tracing     = var.enable_x_ray_tracing
  }
}

# ============================================
# Application Configuration Outputs
# ============================================

output "application_endpoints" {
  description = "Application endpoints for different services"
  value = {
    api_endpoint = var.domain_name != null ? "https://api.${var.domain_name}" : "https://${module.alb.lb_dns_name}"
    health_check = var.domain_name != null ? "https://api.${var.domain_name}/health" : "https://${module.alb.lb_dns_name}/health"
    docs        = var.domain_name != null ? "https://api.${var.domain_name}/docs" : "https://${module.alb.lb_dns_name}/docs"
    monitoring  = var.domain_name != null && var.enable_grafana ? "https://monitoring.${var.domain_name}" : null
  }
}

output "kubernetes_config" {
  description = "Kubernetes configuration for connecting to the cluster"
  value = {
    cluster_name                      = module.eks.cluster_name
    cluster_endpoint                  = module.eks.cluster_endpoint
    cluster_certificate_authority_data = module.eks.cluster_certificate_authority_data
    region                           = data.aws_region.current.name
    aws_auth_config_map             = "aws-auth"
  }
  sensitive = true
}

# ============================================
# Environment-Specific Outputs
# ============================================

output "development_environment" {
  description = "Development environment information"
  value = var.create_dev_environment ? {
    cluster_name = module.dev_environment[0].cluster_name
    vpc_id      = module.dev_environment.vpc_id
    endpoints   = module.dev_environment.endpoints
  } : null
}

output "staging_environment" {
  description = "Staging environment information"  
  value = var.create_staging_environment ? {
    cluster_name = module.staging_environment[0].cluster_name
    vpc_id      = module.staging_environment.vpc_id
    endpoints   = module.staging_environment.endpoints
  } : null
}

# ============================================
# Cost and Resource Information Outputs
# ============================================

output "resource_counts" {
  description = "Count of resources created"
  value = {
    eks_node_groups    = length(var.node_groups)
    rds_instances     = 1 + var.database_read_replica_count
    elasticache_clusters = var.redis_num_cache_clusters
    s3_buckets        = length(var.storage_buckets)
    security_groups   = 4  # eks, rds, elasticache, alb
    subnets          = length(var.public_subnet_cidrs) + length(var.private_subnet_cidrs) + length(var.database_subnet_cidrs)
  }
}

output "estimated_monthly_cost" {
  description = "Estimated monthly cost breakdown (approximate)"
  value = {
    eks_cluster    = "~$73/month (control plane)"
    node_groups    = "Variable based on instance types and count"
    rds_database   = "Variable based on ${var.database_instance_class}"
    elasticache    = "Variable based on ${var.redis_node_type}"
    load_balancer  = "~$23/month"
    s3_storage     = "Variable based on usage"
    data_transfer  = "Variable based on usage"
    note          = "These are rough estimates. Actual costs depend on usage patterns."
  }
}

# ============================================
# Compliance and Governance Outputs
# ============================================

output "compliance_status" {
  description = "Compliance and governance status"
  value = {
    encryption_at_rest     = var.encryption_at_rest_required
    encryption_in_transit  = var.encryption_in_transit_required
    backup_enabled        = true
    monitoring_enabled    = var.enable_cloudwatch_logs
    audit_logging         = var.audit_logging_required
    compliance_frameworks = var.compliance_frameworks
    required_tags_applied = var.resource_tagging_required
  }
}

output "security_features" {
  description = "Security features enabled"
  value = {
    waf_enabled           = var.enable_waf
    shield_advanced       = var.enable_shield_advanced
    guardduty_enabled     = var.enable_guardduty
    config_rules_enabled  = var.enable_config_rules
    cloudtrail_enabled    = var.enable_cloudtrail
    security_hub_enabled  = var.enable_security_hub
    kms_encryption       = true
    secrets_manager      = true
    iam_roles_for_sa     = true
  }
}

# ============================================
# Connection Information Outputs
# ============================================

output "database_connection_string" {
  description = "Database connection string template"
  value       = "postgresql://${var.database_username}:<password>@${module.rds.db_instance_endpoint}:${module.rds.db_instance_port}/${var.database_name}"
  sensitive   = true
}

output "redis_connection_string" {
  description = "Redis connection string template"
  value       = var.redis_auth_token_enabled ? "redis://:<auth-token>@${module.elasticache.primary_endpoint_address}:${var.redis_port}" : "redis://${module.elasticache.primary_endpoint_address}:${var.redis_port}"
  sensitive   = true
}

# ============================================
# Kubectl Configuration Output
# ============================================

output "kubectl_config_command" {
  description = "Command to configure kubectl for the EKS cluster"
  value       = "aws eks --region ${data.aws_region.current.name} update-kubeconfig --name ${module.eks.cluster_name}"
}

output "kubeconfig_filename" {
  description = "Suggested filename for kubeconfig"
  value       = "kubeconfig-${local.cluster_name}"
}

# ============================================
# Operational Commands and Information
# ============================================

output "useful_commands" {
  description = "Useful commands for managing the infrastructure"
  value = {
    # Kubernetes commands
    kubectl_config   = "aws eks --region ${data.aws_region.current.name} update-kubeconfig --name ${module.eks.cluster_name}"
    kubectl_nodes    = "kubectl get nodes"
    kubectl_pods     = "kubectl get pods -A"
    
    # Database commands
    database_connect = "psql -h ${module.rds.db_instance_endpoint} -p ${module.rds.db_instance_port} -U ${var.database_username} -d ${var.database_name}"
    
    # Redis commands
    redis_connect    = "redis-cli -h ${module.elasticache.primary_endpoint_address} -p ${var.redis_port}"
    
    # AWS CLI commands
    describe_cluster = "aws eks describe-cluster --name ${module.eks.cluster_name} --region ${data.aws_region.current.name}"
    list_node_groups = "aws eks list-nodegroups --cluster-name ${module.eks.cluster_name} --region ${data.aws_region.current.name}"
    
    # Monitoring commands
    view_logs       = "aws logs describe-log-groups --region ${data.aws_region.current.name}"
    cloudwatch_insights = "aws logs start-query --region ${data.aws_region.current.name}"
  }
}

output "next_steps" {
  description = "Recommended next steps after infrastructure deployment"
  value = [
    "1. Configure kubectl: ${self.useful_commands.kubectl_config}",
    "2. Deploy Kubernetes applications using the manifests in deployment/kubernetes/",
    "3. Set up monitoring dashboards in Grafana",
    "4. Configure CI/CD pipelines to deploy applications",
    "5. Set up SSL certificates if using custom domains",
    "6. Configure external API keys in AWS Secrets Manager",
    "7. Test application health endpoints",
    "8. Set up alerting rules for monitoring",
    "9. Configure backup schedules",
    "10. Review and update security group rules as needed"
  ]
}

# ============================================
# Troubleshooting Information
# ============================================

output "troubleshooting_info" {
  description = "Information useful for troubleshooting"
  value = {
    common_issues = {
      kubectl_access = "Ensure AWS CLI is configured and run: ${self.useful_commands.kubectl_config}"
      database_access = "Check security groups and ensure connectivity from EKS subnets"
      load_balancer_not_accessible = "Verify security groups, target group health, and DNS configuration"
      pods_not_starting = "Check node capacity, resource requests, and image pull secrets"
    }
    
    log_locations = {
      eks_control_plane = "/aws/eks/${local.cluster_name}/cluster"
      application_logs  = var.enable_cloudwatch_logs ? aws_cloudwatch_log_group.application[0].name : "CloudWatch logs not enabled"
      load_balancer_logs = "Check S3 bucket: ${try(module.s3_buckets["logs"].s3_bucket_id, "logs bucket not configured")}"
    }
    
    monitoring_endpoints = {
      cluster_health = "https://console.aws.amazon.com/eks/home?region=${data.aws_region.current.name}#/clusters/${module.eks.cluster_name}"
      database_monitoring = "https://console.aws.amazon.com/rds/home?region=${data.aws_region.current.name}#database:id=${module.rds.db_instance_identifier}"
      load_balancer_monitoring = "https://console.aws.amazon.com/ec2/v2/home?region=${data.aws_region.current.name}#LoadBalancers"
    }
  }
}

# ============================================
# Tags Applied to Resources
# ============================================

output "common_tags" {
  description = "Common tags applied to all resources"
  value       = local.common_tags
}

output "infrastructure_summary" {
  description = "High-level summary of deployed infrastructure"
  value = {
    project             = var.project_name
    environment         = var.environment
    region              = data.aws_region.current.name
    kubernetes_version  = var.kubernetes_version
    database_engine     = "${var.database_engine} ${var.database_engine_version}"
    high_availability   = var.database_read_replica_count > 0 && var.redis_num_cache_clusters > 1
    encryption_enabled  = var.database_storage_encrypted && var.redis_at_rest_encryption_enabled
    backup_configured   = true
    monitoring_enabled  = var.enable_cloudwatch_logs
    compliance_ready    = var.audit_logging_required && var.encryption_at_rest_required
  }
}
