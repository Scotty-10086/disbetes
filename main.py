# ===================== 环境配置 =====================
# 解决中文显示和错误的核心设置
import matplotlib

matplotlib.use('Agg')  # 使用非交互模式避免GUI报错
import matplotlib.pyplot as plt

# 中文字体配置（Windows系统推荐使用SimHei）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示异常

# ===================== 依赖库导入 =====================
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from pytorch_tabnet.tab_model import TabNetClassifier


# ===================== 数据预处理 =====================
def preprocess_data():
    """
    数据预处理流程：
    1. 加载原始数据并设置列名
    2. 处理数值型异常值（0值替换为中位数）
    3. 类型转换和缺失值处理
    """
    # 加载无列名的原始数据
    df = pd.read_csv('pima-indians-diabetes.data', header=None)

    # 设置正确的列名（根据数据集文档）
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
               'Insulin', 'BMI', 'DiabetesPedigree', 'Age', 'Outcome']
    df.columns = columns

    # 强制转换为数值类型（处理可能的字符串型数值）
    numeric_cols = columns[:-1]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')  # 转换失败设为NaN

    # 处理0值异常（医学指标不可能为0的字段）
    zero_fields = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for field in zero_fields:
        mask = df[field] == 0
        df.loc[mask, field] = np.nan  # 将0值替换为NaN
        median = df[field].median()  # 计算中位数
        df[field] = df[field].fillna(median)  # 中位数填充

    # 清除剩余缺失值并转换标签类型
    df = df.dropna()
    df['Outcome'] = df['Outcome'].astype(int)  # 确保标签为整数类型
    return df


# ===================== 可视化函数 =====================
def plot_confusion_matrix(y_true, y_pred, model_name):
    """绘制混淆矩阵并保存为图片"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"{model_name}混淆矩阵")
    plt.xlabel("预测标签")
    plt.ylabel("真实标签")
    plt.savefig(f"{model_name}_confusion_matrix.png")
    plt.close()


def plot_feature_importance(importances, features, model_name):
    """绘制特征重要性图（适用于树模型）"""
    indices = np.argsort(importances)
    plt.figure(figsize=(10, 6))
    plt.title(f"{model_name}特征重要性")
    plt.barh(range(len(indices)), importances[indices], align="center")
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel("重要性得分")
    plt.savefig(f"{model_name}_feature_importance.png")
    plt.close()


def plot_coefficients(coef, features, model_name):
    """绘制系数绝对值图（适用于线性模型）"""
    indices = np.argsort(np.abs(coef))
    plt.figure(figsize=(10, 6))
    plt.title(f"{model_name}特征系数")
    plt.barh(range(len(indices)), np.abs(coef[indices]), align="center")
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel("系数绝对值")
    plt.savefig(f"{model_name}_coefficients.png")
    plt.close()


# ===================== 主程序 =====================
def main():
    # 数据预处理
    df = preprocess_data()
    print("数据样例：\n", df.head())
    print("\n数据类型：\n", df.dtypes)

    # 特征与标签分离
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # 数据分割（分层抽样保证类别比例）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y  # 保持类别分布
    )

    # 特征标准化（提升模型收敛速度）
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    features = X.columns.tolist()  # 保存特征名称用于可视化

    # 模型定义（包含不同算法）
    models = {
        "逻辑回归": LogisticRegression(max_iter=1000),  # 增加迭代次数确保收敛
        "决策树": DecisionTreeClassifier(max_depth=5),  # 限制深度防止过拟合
        "随机森林": RandomForestClassifier(n_estimators=100),  # 100棵树的森林
        "XGBoost": XGBClassifier(eval_metric="logloss", enable_categorical=False),  # 禁用类别型特征
        "TabNet": TabNetClassifier(
            n_d=16,  # 决策层维度
            n_a=16,  # 注意力层维度
            gamma=1.3  # 正则化系数
        )
    }

    # 模型训练与评估
    for name, model in models.items():
        print(f"\n{'=' * 30}\n正在训练 {name} 模型...")

        try:
            # TabNet特殊处理
            if name == "TabNet":
                model.fit(
                    X_train.astype(np.float32),  # 转换为32位浮点数
                    y_train.values.astype(int).ravel(),  # 确保标签为整数
                    eval_set=[(X_test.astype(np.float32), y_test.values.astype(int).ravel())],
                    max_epochs=100,  # 最大训练轮次
                    patience=20,  # 早停等待轮数
                    batch_size=64,  # 批次大小
                    virtual_batch_size=32,  # 虚拟批次大小
                    eval_metric=['auc']  # 使用AUC作为评估指标
                )
                y_pred = model.predict(X_test.astype(np.float32)).flatten()
            else:
                # 常规模型训练
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

            # 模型评估
            print(f"\n{name} 评估结果:")
            print("准确率:", accuracy_score(y_test, y_pred))
            print("分类报告:\n", classification_report(y_test, y_pred))

            # 生成可视化结果
            plot_confusion_matrix(y_test, y_pred, name)

            # 特征重要性/系数可视化
            if hasattr(model, "feature_importances_"):
                plot_feature_importance(model.feature_importances_, features, name)
            elif hasattr(model, "coef_"):
                plot_coefficients(model.coef_[0], features, name)

        except Exception as e:
            print(f"{name} 训练失败: {str(e)}")


if __name__ == "__main__":
    main()
    print("\n所有模型训练完成！可视化图表已保存至当前目录。")
