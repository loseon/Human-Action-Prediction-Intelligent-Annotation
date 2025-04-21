import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
from imblearn.over_sampling import RandomOverSampler

# ================== 配置区域 ==================
DATA_ROOT = "annotations"  # 标注数据根目录
LABELS = ["drink", "read", "play_with_computer"]  # 要训练的动作标签
SEQUENCE_LENGTH = 30  # 序列长度
NUM_FEATURES = 4  # 每个关键点的特征数 (x, y, z, visibility)
NUM_KEYPOINTS = 13  # 修正后的关键点数量（原12改为13）
BATCH_SIZE = 64
EPOCHS = 10
MODEL_SAVE_PATH = "666.h5"
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15
# =============================================

def load_annotated_data(data_root, labels):
    """加载标注的JSON数据并转换为序列样本（使用滑动窗口）"""
    sequences = []
    label_indices = []

    # 修正后的MediaPipe Pose上半身关键点索引（共13个）
    upper_body_keypoints = [
        0,   # 鼻子
        1, 2, 3, 4,  # 左右眼（1,2）、左右耳（3,4）
        11, 12,  # 左右肩膀
        13, 14,  # 左右手肘
        15, 16,  # 左右手腕
        23, 24   # 左右髋部
    ]

    for label_idx, label in enumerate(labels):
        json_dir = os.path.join(data_root, label, "json")
        if not os.path.exists(json_dir):
            print(f"警告: 目录 {json_dir} 不存在，跳过")
            continue

        frame_files = sorted([f for f in os.listdir(json_dir) if f.endswith('.json')])
        print(f"加载 {label} 类别，文件数量: {len(frame_files)}")

        if len(frame_files) < SEQUENCE_LENGTH:
            print(f"警告: {label} 类别的帧数不足，跳过")
            continue

        # 加载所有关键点数据
        all_kpts = []
        for frame_file in frame_files:
            with open(os.path.join(json_dir, frame_file), 'r') as f:
                try:
                    data = json.load(f)
                    kpts = np.array(data['keypoints'])  # shape (33, 4)
                    upper_body_kpts = kpts[upper_body_keypoints]

                    # 修正标准化方法：使用髋部中心
                    left_hip = kpts[23]
                    right_hip = kpts[24]
                    torso_center = (left_hip[:3] + right_hip[:3]) / 2
                    upper_body_kpts[:, :3] -= torso_center[:3]

                    all_kpts.append(upper_body_kpts)
                except Exception as e:
                    print(f"加载 {frame_file} 时出错: {e}")
                    continue

        # 使用滑动窗口生成序列（步长=1）
        for i in range(len(all_kpts) - SEQUENCE_LENGTH + 1):
            sequence = all_kpts[i:i + SEQUENCE_LENGTH]
            sequences.append(np.array(sequence))
            label_indices.append(label_idx)

    # 处理类别不平衡
    ros = RandomOverSampler(random_state=42)
    sequences_reshaped = np.array(sequences).reshape(len(sequences), -1)
    sequences_resampled, label_indices_resampled = ros.fit_resample(sequences_reshaped, label_indices)

    # 恢复原始形状
    sequences_resampled = sequences_resampled.reshape(-1, SEQUENCE_LENGTH, NUM_KEYPOINTS, NUM_FEATURES)

    # 转换为one-hot编码
    label_vecs = tf.keras.utils.to_categorical(label_indices_resampled, num_classes=len(labels))

    return sequences_resampled, label_vecs


def build_improved_lstm_model(input_shape, num_classes):
    """构建改进的LSTM模型架构"""
    model = Sequential([
        # 输入层
        Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2(0.01)),
                      input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.5),

        # 第二层LSTM
        Bidirectional(LSTM(256, return_sequences=True, kernel_regularizer=l2(0.01))),
        BatchNormalization(),
        Dropout(0.5),

        # 第三层LSTM
        Bidirectional(LSTM(128, return_sequences=False, kernel_regularizer=l2(0.01))),
        BatchNormalization(),
        Dropout(0.5),

        # 全连接层
        Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.5),

        # 输出层
        Dense(num_classes, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy',
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')]
    )
    return model


def main():
    # 1. 加载数据
    X, y = load_annotated_data(DATA_ROOT, LABELS)
    print(f"加载数据完成! 总样本数: {len(X)}, 类别分布: {dict(Counter(np.argmax(y, axis=1)))}")

    # 2. 划分训练集、验证集和测试集
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(VALIDATION_SPLIT + TEST_SPLIT),
        random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=TEST_SPLIT / (VALIDATION_SPLIT + TEST_SPLIT),
        random_state=42, stratify=y_temp
    )

    # 3. 计算类别权重（处理不平衡数据）
    y_integers = np.argmax(y_train, axis=1)
    class_weights = compute_class_weight('balanced', classes=np.unique(y_integers), y=y_integers)
    class_weights = dict(enumerate(class_weights))
    print("类别权重:", class_weights)

    # 4. 构建模型
    input_shape = (SEQUENCE_LENGTH, NUM_KEYPOINTS * NUM_FEATURES)
    model = build_improved_lstm_model(input_shape, len(LABELS))
    model.summary()

    # 5. 训练配置
    callbacks = [
        ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, mode='max'),
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
    ]

    # 6. 训练模型
    history = model.fit(
        X_train.reshape(-1, SEQUENCE_LENGTH, NUM_KEYPOINTS * NUM_FEATURES),
        y_train,
        validation_data=(X_val.reshape(-1, SEQUENCE_LENGTH, NUM_KEYPOINTS * NUM_FEATURES), y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weights,
        shuffle=True
    )

    # 7. 在测试集上评估
    test_loss, test_acc, test_precision, test_recall = model.evaluate(
        X_test.reshape(-1, SEQUENCE_LENGTH, NUM_KEYPOINTS * NUM_FEATURES), y_test
    )
    print(f"\n测试集性能:")
    print(f"准确率: {test_acc:.4f}")
    print(f"精确率: {test_precision:.4f}")
    print(f"召回率: {test_recall:.4f}")

    print(f"训练完成! 最佳模型已保存到: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    # 设置TensorFlow日志级别和GPU配置
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.set_memory_growth(physical_devices[0], True)

    main()
