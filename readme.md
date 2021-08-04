# Mask imposer

Tool to overlay fake face masks.

## Table of contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
    - [Options](#options)
    - [Workflow](#workflow)

## Introduction

## Installation

```shell
pip install git+https://github.com/sqoshi/mask-imposer.git
```

## Usage
`mim INPUT_DIR --option argument`
### Options

| Option | Required | Default | Description |
|:----:|:----:|:----:|:----:|
| input_dir | ✔️ | -- | Input directory. |
| --output-dir | ❌ | results | Output directory. |
| --output-format | ❌ | png | Output images format. |
| --shape-predictor | ❌ | None | Path to shape predictor. |
| --show-samples | ❌ | False | Show sample after detection. |
| --draw-landmarks | ❌ | False | Draw circles on detected landmarks cords. |
| --detect-face-boxes | ❌ | False | Before landmark prediction detect face box. |

### Workflow
