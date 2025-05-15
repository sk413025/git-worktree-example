# 使用 Git Worktree 管理机器学习实验

这个项目演示如何使用 git worktree 来管理多个机器学习实验分支，并比较它们的性能差异。

## 实验介绍

我们实现了四种不同的机器学习实验：
1. **基础模型实验** - 使用默认参数的随机森林分类器
2. **参数调优实验** - 使用网格搜索调整随机森林超参数
3. **特征工程实验** - 对数据进行特征工程处理，包括标准化和多项式特征
4. **模型选择实验** - 比较多种不同的模型算法，选择最佳模型
5. **整合实验** - 结合前三种实验的优势，进行特征工程+参数调优+模型选择的流水线实验

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

# 查看所有工作树
git worktree list
```

## 完整工作流程示例

以下是使用 git worktree 管理机器学习实验的完整工作流程：

1. **创建基础实验**
   ```bash
   # 初始化仓库
   git init
   
   # 添加基础代码并提交
   git add .
   git commit -m "Initial commit with base ML model"
   ```

2. **创建实验分支**
   ```bash
   # 创建超参数调优实验分支
   git checkout -b experiment_hyperparameter
   # 修改代码...
   git commit -am "Add hyperparameter tuning experiment"
   
   # 创建特征工程实验分支
   git checkout master
   git checkout -b experiment_feature_engineering
   # 修改代码...
   git commit -am "Add feature engineering experiment"
   
   # 创建模型选择实验分支
   git checkout master
   git checkout -b experiment_model_selection
   # 修改代码...
   git commit -am "Add model selection experiment"
   
   # 创建整合实验分支
   git checkout master
   git checkout -b experiment_ensemble
   # 修改代码，结合上述三种实验的优势...
   git commit -am "Add ensemble experiment"
   ```

3. **创建工作树**
   ```bash
   git checkout master
   
   # 创建工作树目录
   mkdir -p ../git-worktree-example-experiments
   
   # 为每个实验创建工作树
   git worktree add ../git-worktree-example-experiments/experiment_hyperparameter experiment_hyperparameter
   git worktree add ../git-worktree-example-experiments/experiment_feature_engineering experiment_feature_engineering
   git worktree add ../git-worktree-example-experiments/experiment_model_selection experiment_model_selection
   git worktree add ../git-worktree-example-experiments/experiment_ensemble experiment_ensemble
   
   # 查看工作树列表
   git worktree list
   ```

4. **运行实验**
   ```bash
   # 运行基础实验
   cd /path/to/master/branch
   python train.py
   
   # 运行超参数调优实验
   cd ../git-worktree-example-experiments/experiment_hyperparameter
   python train.py
   
   # 运行特征工程实验
   cd ../experiment_feature_engineering
   python train.py
   
   # 运行模型选择实验
   cd ../experiment_model_selection
   python train.py
   
   # 运行整合实验
   cd ../experiment_ensemble
   python train.py
   ```

5. **比较实验结果**
   ```bash
   cd /path/to/master/branch
   python compare_experiments.py
   ```

## 实验结果比较

实验完成后，我们比较各个实验的性能指标，包括准确率、训练时间等。比较结果会生成一个图表，帮助我们直观地理解不同实验的效果。

## 闭环优化

基于 AlphaEvolve 的思想，我们使用 git worktree 实现了一个闭环优化系统：
1. 从基础模型开始实验
2. 在不同分支上尝试不同优化方法
3. 比较结果，确定有效的优化方向
4. 将多种有效优化方法整合到新的实验中
5. 重复此过程，持续优化模型性能

这种方法可以帮助我们系统地尝试不同方法、跟踪实验结果、合并最佳实践，从而不断改进机器学习模型。

## 优势

使用 git worktree 管理机器学习实验有以下优势：

1. **并行工作** - 可以同时在多个实验上工作，无需频繁切换分支
2. **结果比较** - 所有实验结果可以在文件系统中并排比较
3. **代码隔离** - 每个实验的代码变更互不影响
4. **版本控制** - 所有实验都受到 Git 版本控制的保护
5. **易于协作** - 团队成员可以在不同的实验分支上并行工作
6. **闭环优化** - 通过整合多种实验的优势，不断迭代改进模型 