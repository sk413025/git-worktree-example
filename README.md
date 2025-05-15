# 使用 Git Worktree 管理机器学习实验

这个项目演示如何使用 git worktree 来管理多个机器学习实验分支，并比较它们的性能差异。

## 实验介绍

我们将实现三种不同的机器学习实验：
1. **基础模型实验** - 使用默认参数的随机森林分类器
2. **参数调优实验** - 调整随机森林超参数
3. **特征工程实验** - 对数据进行特征工程处理
4. **模型选择实验** - 使用不同的模型算法

## 如何使用 Git Worktree

Git worktree 允许你同时在同一个仓库的多个分支上工作，每个分支位于不同的目录中。这对于机器学习实验非常有用，可以在不切换分支的情况下比较不同实验结果。

### 基本操作

```bash
# 创建并切换到新分支
git checkout -b experiment_name

# 创建worktree (在新目录中检出不同分支)
git worktree add ../experiment_name experiment_name

# 移除worktree
git worktree remove ../experiment_name
```

## 实验结果比较

实验完成后，我们将比较各个实验的性能指标，包括准确率、训练时间等。 