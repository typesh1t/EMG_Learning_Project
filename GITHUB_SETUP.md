# 如何将项目上传到GitHub

## 🎯 步骤总览

1. 在GitHub创建仓库
2. 连接本地仓库到GitHub
3. 推送代码
4. 验证成功

---

## 📋 详细步骤

### 步骤1：在GitHub创建仓库

1. **登录GitHub**
   - 访问 https://github.com
   - 使用你的账号登录（如果没有账号，先注册一个）

2. **创建新仓库**
   - 点击右上角的 `+` 号
   - 选择 `New repository`

3. **填写仓库信息**
   ```
   Repository name: EMG-Learning-Project
   Description: EMG信号处理学习项目 - 从零开始的完整教程

   设置：
   ☑️ Public（公开，别人可以看到）
   或
   ☐ Private（私有，只有你能看到）

   ⚠️ 重要：不要勾选以下选项（我们已经有这些文件了）：
   ☐ Add a README file
   ☐ Add .gitignore
   ☐ Choose a license
   ```

4. **点击 `Create repository`**

---

### 步骤2：连接本地仓库到GitHub

GitHub会显示一个页面，包含推送代码的命令。

**方法A：使用HTTPS（推荐新手）**

在项目目录下运行：

```bash
cd /home/ubuntu/桌面/高/EMG_Learning_Project

# 添加远程仓库（替换 YOUR_USERNAME 为你的GitHub用户名）
git remote add origin https://github.com/YOUR_USERNAME/EMG-Learning-Project.git

# 推送代码
git push -u origin main
```

**首次推送时会要求输入GitHub用户名和密码/Token**

**方法B：使用SSH（推荐有经验者）**

如果你已经配置了SSH密钥：

```bash
cd /home/ubuntu/桌面/高/EMG_Learning_Project

# 添加远程仓库（替换 YOUR_USERNAME）
git remote add origin git@github.com:YOUR_USERNAME/EMG-Learning-Project.git

# 推送代码
git push -u origin main
```

---

### 步骤3：输入认证信息

#### 如果使用HTTPS：

**从2021年8月起，GitHub不再支持密码认证，需要使用Personal Access Token (PAT)**

1. **创建Personal Access Token**：
   - 访问 https://github.com/settings/tokens
   - 点击 `Generate new token` → `Generate new token (classic)`
   - 填写：
     - Note: `EMG Project`
     - Expiration: `90 days` 或 `No expiration`
     - 勾选权限：`repo` (全部勾选)
   - 点击 `Generate token`
   - **⚠️ 重要**：立即复制Token，离开页面后就看不到了！

2. **使用Token**：
   ```bash
   # 当提示输入密码时，粘贴你的Token（不是密码！）
   Username: YOUR_USERNAME
   Password: ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx (你的Token)
   ```

3. **保存认证信息**（可选，避免每次输入）：
   ```bash
   git config --global credential.helper store
   ```

---

### 步骤4：验证成功

1. **检查推送结果**：
   ```bash
   git log --oneline
   # 应该看到你的提交记录
   ```

2. **访问GitHub仓库**：
   - 打开 `https://github.com/YOUR_USERNAME/EMG-Learning-Project`
   - 你应该看到所有文件和文件夹
   - README.md的内容会自动显示在页面下方

3. **检查内容**：
   - ✅ 可以看到 `docs/` 文件夹
   - ✅ 可以看到 `tools/` 文件夹
   - ✅ 可以看到 README.md 和其他文档
   - ✅ `data/` 文件夹存在但是空的（数据文件被.gitignore忽略了）

---

## 🔄 后续更新项目

当你修改代码或添加新文件后：

```bash
# 1. 查看修改
git status

# 2. 添加修改的文件
git add .

# 3. 提交修改
git commit -m "描述你的修改内容"

# 4. 推送到GitHub
git push
```

**示例**：
```bash
# 完成第1周的代码
git add code/week01_basics/
git commit -m "完成第1周：EMG基础认知教程和示例代码"
git push

# 完成第2周的代码
git add code/week02_device/
git commit -m "完成第2周：设备认识教程和参数计算示例"
git push
```

---

## 📝 推荐的Commit Message格式

使用清晰的提交信息：

```bash
# 功能添加
git commit -m "feat: 添加信号滤波器实现代码"

# Bug修复
git commit -m "fix: 修复数据加载器的路径问题"

# 文档更新
git commit -m "docs: 更新第3周学习指南"

# 代码重构
git commit -m "refactor: 优化特征提取函数性能"

# 完成作业
git commit -m "homework: 完成第6周作业 - 滤波器设计"

# 添加示例
git commit -m "example: 添加实时信号可视化示例"
```

---

## 🌟 美化你的GitHub仓库

### 1. 添加仓库描述和标签

在GitHub仓库页面：
- 点击右上角的 `⚙️ Settings`
- 在 `About` 部分：
  - Website: 你的项目网站（如果有）
  - Topics: 添加标签，如：
    - `emg`
    - `signal-processing`
    - `machine-learning`
    - `bioinformatics`
    - `python`
    - `tutorial`
    - `education`

### 2. 添加README徽章

在 README.md 顶部添加：

```markdown
# EMG肌电信号学习项目

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

> 从零开始的EMG信号处理完整教程
```

### 3. 启用GitHub Pages（可选）

如果想要一个项目网站：
- Settings → Pages
- Source: `Deploy from a branch`
- Branch: `main` → `docs`
- Save

---

## ❓ 常见问题

### Q1: Push被拒绝，提示 "rejected"
```bash
# 先拉取远程更改
git pull origin main --rebase

# 再推送
git push origin main
```

### Q2: 忘记添加.gitignore，已经推送了大文件
```bash
# 从Git历史中删除大文件（谨慎使用）
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch data/large_file.csv" \
  --prune-empty --tag-name-filter cat -- --all

# 强制推送
git push origin main --force
```

### Q3: 想要撤销某个commit
```bash
# 撤销最后一次commit（保留修改）
git reset --soft HEAD~1

# 撤销最后一次commit（丢弃修改）
git reset --hard HEAD~1
```

### Q4: 创建新分支开发新功能
```bash
# 创建并切换到新分支
git checkout -b feature-week03

# 开发完成后，推送分支
git push -u origin feature-week03

# 在GitHub上创建Pull Request合并到main
```

---

## 📊 Git工作流建议

### 简单工作流（推荐初学者）

```
main分支 ← 所有更改直接提交到这里
```

```bash
# 每周学习完成后
git add .
git commit -m "完成第X周学习"
git push
```

### 进阶工作流（推荐有经验者）

```
main分支 ← 稳定版本
    ↑
    └─ dev分支 ← 开发中
           ↑
           ├─ week01 ← 第1周分支
           ├─ week02 ← 第2周分支
           └─ ...
```

```bash
# 开始新一周学习时创建分支
git checkout -b week01

# 学习完成后推送
git push -u origin week01

# 在GitHub创建PR合并到dev
# 定期将dev合并到main
```

---

## 🎓 Git学习资源

- **Git官方教程**: https://git-scm.com/book/zh/v2
- **GitHub入门**: https://guides.github.com/
- **交互式Git教程**: https://learngitbranching.js.org/?locale=zh_CN
- **Git可视化工具**:
  - GitKraken
  - SourceTree
  - GitHub Desktop

---

## ✅ 检查清单

推送前确保：
- [ ] 已创建GitHub仓库
- [ ] 已配置git用户信息（`git config user.name` 和 `user.email`）
- [ ] 已生成Personal Access Token（如果使用HTTPS）
- [ ] 已添加远程仓库（`git remote -v` 查看）
- [ ] 已提交所有修改（`git status` 查看）
- [ ] .gitignore正确配置（大文件不会被上传）

推送后验证：
- [ ] GitHub页面可以看到所有文件
- [ ] README.md正确显示
- [ ] 文档可以正常浏览
- [ ] 没有意外上传大文件或敏感信息

---

## 🎉 完成！

如果一切顺利，你的项目现在已经在GitHub上了！

**项目地址**：`https://github.com/YOUR_USERNAME/EMG-Learning-Project`

可以分享给：
- 🎓 同学和朋友
- 👨‍🏫 导师
- 📝 大学申请材料
- 💼 求职简历

---

**祝你的GitHub之旅顺利！** 🚀

有任何问题，请查阅Git文档或询问导师。
