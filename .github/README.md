# CI/CD Pipeline Configuration

This directory contains GitHub Actions workflows for continuous integration and deployment of the audio-notes project.

## Workflows Overview

### 🔄 CI Pipeline (`ci.yml`)

**Triggers:** Push to `main`/`develop`, Pull Requests, Manual dispatch

**Jobs:**
- **Code Quality & Linting**: Runs Black, isort, Flake8, and MyPy checks
- **Test Suite**: Cross-platform testing (Ubuntu, Windows, macOS) with Python 3.12 & 3.13
- **Ollama Integration Tests**: Tests AI-powered features (limited to main branch)
- **Build Package**: Validates package building and structure
- **Security Scan**: Runs safety and bandit security tools

**Features:**
- ✅ Multi-platform testing
- ✅ Parallel test execution
- ✅ Caching for faster builds
- ✅ Comprehensive code quality checks
- ✅ Security vulnerability scanning

### 🚀 CD Pipeline (`cd.yml`)

**Triggers:** GitHub releases, Manual dispatch

**Jobs:**
- **Validate Release**: Ensures version consistency and quality
- **Build & Test Package**: Cross-platform package building and testing
- **Publish to Test PyPI**: Safe testing of package publishing
- **Publish to PyPI**: Production package release
- **GitHub Release Assets**: Automated release asset management

**Features:**
- ✅ Automated version validation
- ✅ Test PyPI staging
- ✅ Cross-platform package testing
- ✅ Release asset generation
- ✅ Checksum generation

### 🔐 Dependency Management (`dependency-update.yml`)

**Triggers:** Weekly schedule (Mondays 9 AM UTC), Manual dispatch

**Jobs:**
- **Security Audit**: Regular vulnerability scanning
- **Dependency Review**: PR-based dependency change analysis
- **Update Dependencies**: Automated dependency updates
- **License Check**: License compliance monitoring

**Features:**
- ✅ Weekly automated updates
- ✅ Security vulnerability detection
- ✅ License compliance checking
- ✅ Automated PR creation for updates

### ⚡ Performance Testing (`performance.yml`)

**Triggers:** Weekly schedule (Sundays 6 AM UTC), PR changes, Manual dispatch

**Jobs:**
- **Performance Benchmarks**: Audio processing performance testing
- **Stress Testing**: Large file processing validation

**Features:**
- ✅ Real-time factor analysis
- ✅ Memory usage monitoring
- ✅ Performance regression detection
- ✅ Automated performance reports

## Setup Requirements

### Repository Secrets

For the CD pipeline to work properly, configure these secrets in your GitHub repository:

#### PyPI Publishing
```
TEST_PYPI_API_TOKEN  # Token for test.pypi.org
PYPI_API_TOKEN       # Token for pypi.org
```

#### Environments

Create the following GitHub environments in your repository settings:

1. **test-pypi**
   - Required reviewers: (optional)
   - Protection rules: (optional)
   - Secrets: `TEST_PYPI_API_TOKEN`

2. **pypi**
   - Required reviewers: (recommended for production)
   - Protection rules: Require main branch
   - Secrets: `PYPI_API_TOKEN`

### API Token Setup

#### PyPI Tokens
1. Go to [PyPI Account Settings](https://pypi.org/manage/account/)
2. Create API token with project scope
3. Add token to GitHub repository secrets

#### Test PyPI Tokens
1. Go to [Test PyPI Account Settings](https://test.pypi.org/manage/account/)
2. Create API token with project scope
3. Add token to GitHub repository secrets

## Workflow Configuration

### Python Version Strategy

- **Default Version**: Python 3.12
- **Test Matrix**: Python 3.12, 3.13
- **Minimum Supported**: Python 3.12 (as specified in pyproject.toml)

### Platform Support

- **Primary**: Ubuntu Latest (fastest, most features)
- **Secondary**: Windows Latest, macOS Latest (compatibility testing)

### Caching Strategy

- **uv cache**: Enabled for dependency installation speedup
- **Build artifacts**: Retained for 7-30 days
- **Test results**: Available in workflow runs

## Usage Examples

### Manual Release Workflow

1. **Prepare Release**
   ```bash
   # Update version in pyproject.toml
   git tag v0.2.0
   git push origin v0.2.0
   ```

2. **Create GitHub Release**
   - Go to GitHub releases
   - Create release from tag
   - CD workflow automatically triggers

### Manual Dependency Update

1. **Trigger Workflow**
   - Go to Actions tab
   - Select "Dependency Updates & Security"
   - Click "Run workflow"

2. **Review Generated PR**
   - Automated PR created with dependency updates
   - Review changes and merge if appropriate

### Performance Testing

1. **On PR**: Automatic performance benchmarks for code changes
2. **Manual**: Trigger via Actions tab for ad-hoc testing
3. **Weekly**: Automated performance monitoring

## Monitoring & Maintenance

### Regular Tasks

- **Weekly**: Review dependency update PRs
- **Monthly**: Review security scan results
- **Quarterly**: Update workflow actions to latest versions

### Performance Baselines

- **Target Real-time Factor**: > 1.0x for audio processing
- **Memory Usage**: Stable across file sizes
- **Test Coverage**: Maintain > 80%

### Troubleshooting

#### CI Failures

1. **Lint Failures**: Run `uv run black src/ tests/` and `uv run isort src/ tests/`
2. **Test Failures**: Check test logs and fix failing tests
3. **Build Failures**: Verify pyproject.toml configuration

#### CD Failures

1. **Version Mismatch**: Ensure git tag matches pyproject.toml version
2. **PyPI Issues**: Verify API tokens and permissions
3. **Test PyPI**: Check if package already exists with same version

#### Performance Regressions

1. **Check Benchmarks**: Review performance report artifacts
2. **Compare Baselines**: Look for significant changes in real-time factor
3. **Memory Leaks**: Monitor memory usage patterns

## Contributing

When adding new workflows or modifying existing ones:

1. **Test Locally**: Use `act` or similar tools when possible
2. **Start Small**: Test with simple changes first
3. **Document Changes**: Update this README
4. **Monitor Results**: Watch first few runs carefully

## Advanced Configuration

### Custom Test Markers

The pytest configuration supports these markers:
- `unit`: Unit tests (fast)
- `integration`: Integration tests
- `ollama`: AI-powered tests (require external service)
- `slow`: Long-running tests
- `network`: Tests requiring internet access

### Workflow Customization

Each workflow can be customized by:
- Modifying trigger conditions
- Adjusting Python version matrix
- Adding new test categories
- Changing artifact retention policies

### Security Considerations

- All secrets are properly scoped
- Workflows use pinned action versions
- Minimal permissions principle applied
- Regular security scanning enabled