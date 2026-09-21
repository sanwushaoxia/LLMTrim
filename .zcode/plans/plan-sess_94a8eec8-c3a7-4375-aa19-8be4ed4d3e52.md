在项目根目录创建 `/home/wyh/Automation/Llmtrim/.gitignore`，覆盖本项目实际产生的及常见的缓存/临时文件：

1. **Python 字节码缓存**：`__pycache__/`、`*.py[cod]`、`*$py.class`
2. **打包构建产物**：`*.egg-info/`（本项目 pip install -e 产生的 `llmtrim.egg-info/`）、`build/`、`dist/`、`.eggs/`
3. **测试/覆盖率缓存**：`.pytest_cache/`、`.coverage`、`htmlcov/`、`.tox/`
4. **虚拟环境**：`.venv/`、`venv/`、`env/`
5. **工具缓存**：`.mypy_cache/`、`.ruff_cache/`、`.ipynb_checkpoints/`
6. **编辑器/系统文件**：`.vscode/`、`.idea/`、`.DS_Store`

仅创建这一个文件，不修改任何其他内容，不执行 git 操作。