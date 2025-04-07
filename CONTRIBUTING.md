# Contributing to Arthashila

Thank you for your interest in contributing to Arthashila! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please be respectful and considerate of others.

## How to Contribute

### 1. Fork the Repository
- Fork the repository to your GitHub account
- Clone your fork to your local machine
```bash
git clone https://github.com/yourusername/arthashila.git
cd arthashila
```

### 2. Set Up Development Environment
- Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
- Install development dependencies
```bash
pip install -r requirements.txt
```

### 3. Create a Feature Branch
- Create a new branch for your feature or bugfix
```bash
git checkout -b feature/your-feature-name
```

### 4. Make Your Changes
- Follow the project's coding style (see below)
- Write clear, concise commit messages
- Include tests for new features
- Update documentation as needed

### 5. Test Your Changes
- Run the application locally
```bash
python -m streamlit run main.py
```
- Ensure all tests pass
- Check for any linting errors

### 6. Submit a Pull Request
- Push your changes to your fork
```bash
git push origin feature/your-feature-name
```
- Create a Pull Request from your fork to the main repository
- Provide a clear description of your changes
- Reference any related issues

## Coding Style

### Python Style Guide
- Follow PEP 8 guidelines
- Use type hints where appropriate
- Write docstrings for all functions and classes
- Keep functions focused and single-purpose

### Documentation
- Update README.md for major changes
- Add comments to explain complex logic
- Document new features in the appropriate files

### Testing
- Write unit tests for new features
- Ensure existing tests pass
- Include test cases for edge cases

## Project Structure

```
arthashila/
├── features/          # Feature modules
├── utils/            # Utility functions
├── data/             # Data files
├── main.py           # Main application
├── requirements.txt  # Dependencies
└── tests/            # Test files
```

## Feature Development

When adding new features:
1. Create a new module in the `features` directory
2. Implement the feature following the existing patterns
3. Add appropriate imports in `features/__init__.py`
4. Update the main application to include the new feature
5. Add tests for the new functionality

## Bug Reports

When reporting bugs:
1. Check if the issue has already been reported
2. Provide detailed steps to reproduce
3. Include system information and error messages
4. Suggest possible solutions if you have any

## Feature Requests

When requesting features:
1. Check if the feature has already been requested
2. Explain why the feature would be useful
3. Provide examples of how it would work
4. Consider contributing the feature yourself

## Documentation

When updating documentation:
1. Keep it clear and concise
2. Use proper formatting
3. Include examples where helpful
4. Update all related documentation

## Questions?

If you have any questions about contributing:
- Open an issue in the repository
- Contact the maintainers
- Join our community discussions

Thank you for contributing to Arthashila! 