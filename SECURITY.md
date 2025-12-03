# Security Policy

## Supported Versions

We release security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.x.x   | :white_check_mark: |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to: **security@cloudopt-ai.example**

You should receive a response within 48 hours. If for some reason you do not, please follow up via email to ensure we received your original message.

Please include the following information:

- Type of issue (e.g., buffer overflow, SQL injection, cross-site scripting, etc.)
- Full paths of source file(s) related to the manifestation of the issue
- The location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

## Security Best Practices

### Secrets Management
- **Never** commit secrets, API keys, or credentials to the repository
- Use environment variables for sensitive configuration
- Utilize secret management systems (AWS Secrets Manager, HashiCorp Vault)
- Rotate credentials regularly

### Dependencies
- Keep dependencies up to date
- Run `npm audit` and `pip-audit` regularly
- Review security advisories from GitHub Dependabot
- Use lock files to ensure reproducible builds

### Authentication & Authorization
- Implement proper authentication for all APIs
- Use OAuth2/OIDC for user authentication
- Apply principle of least privilege
- Validate all user inputs

### Infrastructure Security
- Enable encryption at rest and in transit
- Use security groups and network policies
- Implement logging and monitoring
- Regular security scanning with Trivy, Snyk
- IaC security validation with checkov, tfsec

### CI/CD Security
- Scan container images for vulnerabilities
- Use signed commits
- Implement branch protection rules
- Secret scanning in CI pipeline (gitleaks)
- SAST/DAST integration

## Security Scanning

### Pre-commit Hooks
```bash
# Install pre-commit hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

### Local Scanning
```bash
# Scan for secrets
make scan-secrets

# Scan IaC
make scan-iac

# Scan dependencies
make scan-deps

# Scan container images
make scan-images
```

### CI Pipeline
All pull requests automatically run:
- Gitleaks (secret scanning)
- Trivy (container & dependency scanning)
- Checkov (IaC security)
- Snyk (dependency vulnerabilities)

## Vulnerability Disclosure Policy

We believe in responsible disclosure and will work with security researchers to:

1. **Acknowledge** receipt of vulnerability report within 48 hours
2. **Confirm** the vulnerability and determine severity within 7 days
3. **Develop** a fix and timeline for release
4. **Release** security patch
5. **Publish** security advisory (after fix is available)

### Recognition
We maintain a [Hall of Fame](docs/SECURITY_HALL_OF_FAME.md) to recognize security researchers who responsibly disclose vulnerabilities.

## Security Updates

Security advisories are published at:
- GitHub Security Advisories
- Security mailing list: security-announce@cloudopt-ai.example

Subscribe to stay informed about security updates.

## Contact

- **Security Email**: security@cloudopt-ai.example
- **PGP Key**: [Download PGP key](docs/pgp-key.asc)

---

Thank you for helping keep CloudOpt AI and our users safe!
