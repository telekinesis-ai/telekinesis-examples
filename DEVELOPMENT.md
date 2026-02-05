# Development Guide

This guide is for contributors working on the `telekinesis-examples` repository.

## Table of Contents

- [Getting Started](#getting-started)
- [Working with Git Submodules](#working-with-git-submodules)
- [Development Workflow](#development-workflow)
- [Adding New Examples](#adding-new-examples)

## Getting Started

### Prerequisites

- Python 3.11 or 3.12
- Git
- A Telekinesis API key from [platform.telekinesis.ai](https://platform.telekinesis.ai/api-keys)

### Initial Setup

1. Clone the repository with submodules:

```bash
git clone --recurse-submodules https://github.com/telekinesis-ai/telekinesis-examples.git
cd telekinesis-examples
```

2. If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

3. Install dependencies:

```bash
pip install telekinesis-ai
pip install numpy scipy opencv-python rerun-sdk==0.27.3 loguru
```

4. Set up your API key:

```bash
# macOS/Linux
export TELEKINESIS_API_KEY="your_api_key"

# Windows
setx TELEKINESIS_API_KEY "your_api_key"
```

## Working with Git Submodules

This repository uses the `telekinesis-data` submodule to store sample data (images, point clouds, meshes). Understanding how to work with submodules is crucial for contributors.

### Understanding Submodules

Git submodules are **snapshot references** to specific commits in another repository. When you clone `telekinesis-examples`, it points to a specific commit hash in the `telekinesis-data` repository—not to the latest commit or a branch.

### Checking Submodule Status

To see the current state of the submodule:

```bash
git submodule status
```

- No prefix: submodule is at the expected commit
- `+` prefix: submodule has uncommitted changes (checked out to a different commit than expected)
- `-` prefix: submodule is not initialized

### Updating the Submodule to Latest Remote Commit

When new data is added to the `telekinesis-data` repository, you need to update the submodule reference in `telekinesis-examples`:

```bash
# Update the submodule to the latest commit from remote
git submodule update --remote telekinesis-data

# Stage the submodule update
git add telekinesis-data

# Commit the update
git commit -m "chore: update telekinesis-data submodule to latest"

# Push to remote
git push origin develop  # or main, depending on your branch
```

**Important**: This update must be done **every time** new data is added to `telekinesis-data` that you want to include in `telekinesis-examples`. The submodule reference is a snapshot, not a dynamic link.

### Manually Updating to a Specific Commit

If you need to point to a specific commit (not the latest):

```bash
cd telekinesis-data
git fetch
git checkout <specific-commit-hash>
cd ..
git add telekinesis-data
git commit -m "chore: update telekinesis-data to commit <hash>"
git push origin develop
```

### Common Submodule Issues

**Issue**: Submodule shows as modified but you didn't change anything
```bash
# Reset to the expected commit
git submodule update --init --recursive
```

**Issue**: Submodule is in "detached HEAD" state
- This is normal! Submodules are always in detached HEAD state since they point to specific commits.

**Issue**: After pulling changes, submodule is out of sync
```bash
# Update submodules to match the commit referenced in the parent repo
git submodule update --init --recursive
```

## Development Workflow

### Branching Strategy

- `main`: Stable, production-ready code
- `develop`: Active development branch
- Feature branches: `feature/<feature-name>`

### Creating a Feature Branch

```bash
git checkout develop
git pull origin develop
git checkout -b feature/your-feature-name
```

### Before Committing

1. Ensure all examples run successfully
2. Check code formatting
3. Verify submodule is at the correct commit
4. Update documentation if needed

### Submitting Changes

1. Push your feature branch:
```bash
git push origin feature/your-feature-name
```

2. Create a Pull Request to `develop`

3. Address review feedback

4. Once approved, changes will be merged to `develop` and eventually to `main`

## Adding New Examples

### File Structure

Place new examples in the `examples/` directory:

```
examples/
├── datatypes_examples.py
├── vitreous_examples.py
├── pupil_examples.py
└── your_new_examples.py  # Your new file
```

### Example Template

```python
"""
Examples demonstrating [Skill Group] Skills from the Telekinesis library.
"""

import argparse
from pathlib import Path
from telekinesis import your_skill_group
from loguru import logger

# Data directory
DATA_DIR = Path(__file__).parent.parent / "telekinesis-data"

def example_skill_name():
    """
    Demonstrates the use of skill_name.
    
    Docs: https://docs.telekinesis.ai/...
    """
    logger.info("Running example: skill_name")
    
    # Your example code here
    
    logger.success("Example completed successfully")

# Add to available examples dictionary
EXAMPLES = {
    "skill_name": example_skill_name,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Your Skill Group Examples")
    parser.add_argument("--list", action="store_true", help="List all examples")
    parser.add_argument("--example", type=str, help="Run a specific example")
    
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable examples:")
        for name in EXAMPLES:
            print(f"  - {name}")
    elif args.example:
        if args.example in EXAMPLES:
            EXAMPLES[args.example]()
        else:
            print(f"Example '{args.example}' not found")
    else:
        parser.print_help()
```

### Adding Test Data

If your example requires new test data:

1. Add data to the appropriate directory in the `telekinesis-data` repository
2. Push changes to `telekinesis-data`
3. Update the submodule reference in `telekinesis-examples` (see [Working with Git Submodules](#working-with-git-submodules))
