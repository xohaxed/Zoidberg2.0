# Contributing to CloudOpt AI

Thank you for your interest in contributing to CloudOpt AI! This document provides guidelines and instructions for contributing.

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker & Docker Compose
- Git
- pre-commit

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/cloudopt-ai/cloudopt-ai.git
cd cloudopt-ai

# Install pre-commit hooks
make setup-dev

# Start local development environment
make dev-up

# Run tests
make test
```

## 📝 Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Changes

- Write code following our style guidelines
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass locally

### 3. Commit Changes

We use [Conventional Commits](https://www.conventionalcommits.org/):

```bash
git commit -m "feat: add new recommendation algorithm"
git commit -m "fix: resolve API authentication issue"
git commit -m "docs: update architecture diagram"
git commit -m "test: add integration tests for optimizer"
git commit -m "refactor: simplify data pipeline logic"
```

**Commit Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks
- `ci`: CI/CD changes

### 4. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## 🧪 Testing Requirements

All contributions must include appropriate tests:

### Python Services
```bash
# Run unit tests
pytest services/api/tests/
pytest services/ai-engine/tests/

# Check coverage (minimum 80%)
pytest --cov=app --cov-report=html
```

### TypeScript/React
```bash
# Run tests
npm test

# Check coverage
npm run test:coverage
```

### Infrastructure
```bash
# Validate Terraform
terraform validate

# Run security scans
make scan-iac
```

## 🎨 Code Style

### Python
- Follow PEP 8
- Use Black for formatting
- Use isort for import sorting
- Use mypy for type checking
- Maximum line length: 100

```bash
# Format code
black .
isort .

# Type check
mypy services/
```

### TypeScript/JavaScript
- Follow Airbnb style guide
- Use Prettier for formatting
- Use ESLint for linting

```bash
# Format code
npm run format

# Lint
npm run lint
```

### Infrastructure as Code
- Use consistent naming conventions
- Add comments for complex logic
- Include examples in module READMEs
- Run tflint and checkov

## 📋 Pull Request Guidelines

### PR Checklist

- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] All tests passing
- [ ] No new warnings
- [ ] Conventional commit messages used

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
Describe testing performed

## Screenshots (if applicable)

## Related Issues
Fixes #123
```

## 🏗️ Architecture Guidelines

### Service Design
- Follow microservices best practices
- Use dependency injection
- Implement proper error handling
- Add structured logging
- Include health check endpoints

### API Design
- RESTful conventions
- OpenAPI/Swagger documentation
- Versioned endpoints (e.g., `/api/v1/`)
- Consistent error responses
- Rate limiting considerations

### Database
- Use migrations (Alembic)
- Index frequently queried fields
- Document schema changes
- Consider backward compatibility

### Security
- Never commit secrets
- Use environment variables
- Validate all inputs
- Implement proper authentication/authorization
- Follow OWASP guidelines

## 📚 Documentation

### Code Documentation
- Docstrings for all public functions/classes
- Type hints in Python
- JSDoc comments in TypeScript
- README in each service directory

### Architecture Decisions
- Document significant decisions in ADRs
- Location: `docs/decision-records/`
- Template available in `docs/decision-records/template.md`

## 🔍 Review Process

1. Automated checks run (CI/CD)
2. Code review by maintainer(s)
3. Address feedback
4. Approval and merge

### Review Criteria
- Code quality and style
- Test coverage
- Documentation completeness
- Performance considerations
- Security implications

## 🐛 Bug Reports

Use GitHub Issues with the bug report template:

- Clear title
- Steps to reproduce
- Expected vs actual behavior
- Environment details
- Screenshots/logs if applicable

## 💡 Feature Requests

Use GitHub Issues with the feature request template:

- Clear description
- Use case/motivation
- Proposed solution
- Alternatives considered

## 📞 Communication

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: Questions, ideas
- **Pull Requests**: Code contributions
- **Email**: security@cloudopt-ai.example (security issues only)

## 🎓 Learning Resources

- [Project Architecture](docs/architecture.md)
- [API Documentation](services/api/README.md)
- [Onboarding Guide](docs/onboarding.md)
- [Decision Records](docs/decision-records/)

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to CloudOpt AI! 🙏
