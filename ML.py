import numpy as np
import os
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from mpl_toolkits.axes_grid1 import make_axes_locatable

def load_dataset(folder_path, lag=1, window_size=3):
    data_list = []
    label_list = []
    label_list_org = []
    data_list2 = []
    label_list2 = []
    crop_type_labels = []
    crop_list = []
    crop_list2 = []
    files = sorted(os.listdir(folder_path))
    
    # Determine the maximum dimensions for X arrays
    max_rows = 0
    max_cols = 0
    for file in files:
        if file.endswith("_data.npy"):
            data_path = os.path.join(folder_path, file)
            data = np.load(data_path)
            combined_rows = data.shape[0] * data.shape[1]
            combined_cols = data.shape[2] * data.shape[3]
            max_rows = max(max_rows, combined_rows)
            max_cols = max(max_cols, combined_cols)

    for file in files:
        if file.endswith("_data.npy"):
            data_path = os.path.join(folder_path, file)
            data = np.load(data_path)
            data = data[:-1, :, :, :]
            X = data.reshape(data.shape[0] * data.shape[1], data.shape[2] * data.shape[3])
            X = np.transpose(X)
            padded_X = np.zeros((max_cols, max_rows))
            padded_X2 = np.zeros((X.shape[0], max_rows))
            padded_X[:X.shape[0], :X.shape[1]] = X
            padded_X2[:, :X.shape[1]] = X
            data_list.append(padded_X)
            data_list2.append(padded_X2)

            label_file = file.replace("_data.npy", "_yield.npy")
            label_path = os.path.join(folder_path, label_file)
            if os.path.exists(label_path):
                label = np.load(label_path)
                crop_type = file.split('_')[1]
                crop_list.append(crop_type)

                if crop_type == "corn":
                    label *= 12.454
                elif crop_type in ["soy", "wheat"]:
                    label *= 13.34368

                if label.shape[0] > 2 and label.shape[1] > 2:
                    label[0:1, :] = 0
                    label[:, 0:1] = 0
                    label[:, -1] = 0
                    label[-1, :] = 0

                if label.shape[0] <= 2 and label.shape[1] > 2:
                    label[:, 0:1] = 0
                    label[:, -1] = 0

                if label.shape[0] > 2 and label.shape[1] <= 2:
                    label[0:1, :] = 0
                    label[-1, :] = 0

                label_list_org.append(label)
                Y = label.reshape(label.shape[0] * label.shape[1])
                crop = np.full(label.shape[0] * label.shape[1], crop_type)
                padded_Y = np.zeros(max_cols)
                padded_Y[:Y.shape[0]] = Y
                label_list.append(padded_Y)
                crop_type_array = np.full(padded_Y.shape, crop_type)
                crop_type_labels.append(crop_type_array)
                crop_list2.append(crop)
                label_list2.append(Y)

    crop_type_encoder = LabelEncoder()
    crop_type_labels_flattened = [item for sublist in crop_type_labels for item in sublist.ravel()]
    crop_type_labels_encoded = crop_type_encoder.fit_transform(crop_type_labels_flattened)
    crop_type_labels_encoded2 = crop_type_encoder.fit_transform(crop_type_array)

    data_array = np.concatenate(data_list, axis=0)
    labels_array = np.concatenate(label_list, axis=0)

    return data_array, labels_array, crop_type_labels_encoded, data_list, label_list, data_list2, label_list2, label_list_org, crop_list, crop_list2

def evaluate_classification_model(model, X_test, test_crop_labels):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(test_crop_labels, y_pred)
    precision = precision_score(test_crop_labels, y_pred, average='macro')
    recall = recall_score(test_crop_labels, y_pred, average='macro')
    f1 = f1_score(test_crop_labels, y_pred, average='macro')
    conf_matrix = confusion_matrix(test_crop_labels, y_pred)
    return accuracy, precision, recall, f1, conf_matrix

def evaluate_model(model, X_test, test_yield):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(test_yield, y_pred)
    rmse = np.sqrt(mean_squared_error(test_yield, y_pred))
    r2 = r2_score(test_yield, y_pred)
    NRMSE = np.sqrt(mean_squared_error(test_yield, y_pred)) / (np.max(test_yield) - np.min(test_yield))
    ME = 1 - (np.sum((test_yield - y_pred) ** 2) / np.sum((test_yield - np.mean(test_yield)) ** 2))
    d = 1 - (np.sum((test_yield - y_pred) ** 2) / np.sum((np.abs(y_pred - np.mean(test_yield)) + np.abs(test_yield - np.mean(test_yield))) ** 2))
    mape = np.mean(np.abs((test_yield - y_pred) / test_yield)) * 100
    return mae, rmse, r2, NRMSE, ME, d, mape

start_time = time.time()

# Load datasets
folder_path_train = 'training_dataset'
folder_path_test = 'test_dataset'
train_data, train_yield, train_crop_labels, data_list, label_list, data_list2, label_list2, label_list_org, crop_list, crop_list2 = load_dataset(folder_path_train)
test_data, test_yield, test_crop_labels, data_list, label_list, data_list2, label_list2, label_list_org, crop_list, crop_list2 = load_dataset(folder_path_test)

train_data = np.array(train_data)
train_yield = np.array(train_yield)
train_crop_labels = np.array(train_crop_labels)
label_list2 = np.array(label_list2)
crop_list2 = np.array(crop_list2)

print('train_data', np.shape(train_data))
print('train_yield', np.shape(train_yield))
print('train_crop', np.shape(train_crop_labels))

test_data = np.array(test_data)
test_labels = np.array(test_yield)
test_crop_labels = np.array(test_crop_labels)

print('test_data', np.shape(test_data))
print('test_labels', np.shape(test_yield))

non_zero_indices = [i for i, label in enumerate(train_yield) if not np.any(label == 0)]
train_data = [train_data[i] for i in non_zero_indices]
train_yield = [train_yield[i] for i in non_zero_indices]
train_crop_labels = [train_crop_labels[i] for i in non_zero_indices]

non_zero_indices = [i for i, label in enumerate(test_yield) if not np.any(label == 0)]
test_data = [test_data[i] for i in non_zero_indices]
test_yield = [test_labels[i] for i in non_zero_indices]
test_crop_labels = [test_crop_labels[i] for i in non_zero_indices]

# Initialize models with given parameters
rf = RandomForestRegressor(n_estimators=400, max_depth=40, min_samples_split=2, min_samples_leaf=1, bootstrap=True)
svm = SVR(C=10, gamma=0.01, kernel='linear')
xgb_model = xgb.XGBRegressor(n_estimators=400, learning_rate=0.1, max_depth=8, colsample_bytree=0.7, subsample=0.9)

# Train the models
rf.fit(train_data, train_yield)
svm.fit(train_data, train_yield)
xgb_model.fit(train_data, train_yield)

# Evaluate Random Forest
mae_rf_train, rmse_rf_train, r2_rf_train, NRMSE_rf_train, ME_rf_train, d_rf_train, mape_rf_train = evaluate_model(rf, train_data, train_yield)
mae_rf_test, rmse_rf_test, r2_rf_test, NRMSE_rf_test, ME_rf_test, d_rf_test, mape_rf_test = evaluate_model(rf, test_data, test_yield)

print(f"Random Forest (Training) - MAE: {mae_rf_train:.3f}, RMSE: {rmse_rf_train:.3f}, R2: {r2_rf_train:.3f}, NRMSE: {NRMSE_rf_train:.3f}, ME: {ME_rf_train:.3f}, d: {d_rf_train:.3f}, MAPE: {mape_rf_train:.3f}")
print(f"Random Forest (Test) - MAE: {mae_rf_test:.3f}, RMSE: {rmse_rf_test:.3f}, R2: {r2_rf_test:.3f}, NRMSE: {NRMSE_rf_test:.3f}, ME: {ME_rf_test:.3f}, d: {d_rf_test:.3f}, MAPE: {mape_rf_test:.3f}")

# Evaluate SVM
mae_svm_train, rmse_svm_train, r2_svm_train, NRMSE_svm_train, ME_svm_train, d_svm_train, mape_svm_train = evaluate_model(svm, train_data, train_yield)
mae_svm_test, rmse_svm_test, r2_svm_test, NRMSE_svm_test, ME_svm_test, d_svm_test, mape_svm_test = evaluate_model(svm, test_data, test_yield)

print(f"SVM (Training) - MAE: {mae_svm_train:.3f}, RMSE: {rmse_svm_train:.3f}, R2: {r2_svm_train:.3f}, NRMSE: {NRMSE_svm_train:.3f}, ME: {ME_svm_train:.3f}, d: {d_svm_train:.3f}, MAPE: {mape_svm_train:.3f}")
print(f"SVM (Test) - MAE: {mae_svm_test:.3f}, RMSE: {rmse_svm_test:.3f}, R2: {r2_svm_test:.3f}, NRMSE: {NRMSE_svm_test:.3f}, ME: {ME_svm_test:.3f}, d: {d_svm_test:.3f}, MAPE: {mape_svm_test:.3f}")

# Evaluate XGBoost
mae_xgb_train, rmse_xgb_train, r2_xgb_train, NRMSE_xgb_train, ME_xgb_train, d_xgb_train, mape_xgb_train = evaluate_model(xgb_model, train_data, train_yield)
mae_xgb_test, rmse_xgb_test, r2_xgb_test, NRMSE_xgb_test, ME_xgb_test, d_xgb_test, mape_xgb_test = evaluate_model(xgb_model, test_data, test_yield)

print(f"XGBoost (Training) - MAE: {mae_xgb_train:.3f}, RMSE: {rmse_xgb_train:.3f}, R2: {r2_xgb_train:.3f}, NRMSE: {NRMSE_xgb_train:.3f}, ME: {ME_xgb_train:.3f}, d: {d_xgb_train:.3f}, MAPE: {mape_xgb_train:.3f}")
print(f"XGBoost (Test) - MAE: {mae_xgb_test:.3f}, RMSE: {rmse_xgb_test:.3f}, R2: {r2_xgb_test:.3f}, NRMSE: {NRMSE_xgb_test:.3f}, ME: {ME_xgb_test:.3f}, d: {d_xgb_test:.3f}, MAPE: {mape_xgb_test:.3f}")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time taken: {elapsed_time:.2f} seconds")

def get_crop_name(crop_num):
    if crop_num == 0:
        return "Corn"
    elif crop_num == 1:
        return "Soy Bean"
    elif crop_num == 2:
        return "Wheat"
    else:
        return "Unknown"

# Plotting
fig, axs = plt.subplots(1, len(crop_list), figsize=(80, 15), dpi=300)
cmap = 'RdYlBu_r'

max_width = max_height = 0
for patch in data_list2:
    height, width = patch.shape[:2]
    max_width = max(max_width, width)
    max_height = max(max_height, height)

extent = [0, max_width, max_height, 0]

columns = ['Corn', 'Soy', 'Wheat']
years = ['2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']

fig, axs = plt.subplots(len(years), len(columns), figsize=(15, 40), dpi=300)
for i, col_name in enumerate(columns):
    axs[0, i].set_title(col_name, fontsize=16)

y_pred_xgb_patch_list = []
mean_diff_list2 = []
for idx, patch in enumerate(data_list2):
    y_pred_xgb_patch = xgb_model.predict(patch)
    test_diffs_xgb_patch = np.absolute(y_pred_xgb_patch - label_list2[idx])
    y_pred_xgb_patch_list.append(np.mean(y_pred_xgb_patch[label_list2[idx] != 0]))
    mean_diff_list2.append(np.mean(label_list2[idx][label_list2[idx] != 0]))
    test_diffs_xgb_patch[label_list2[idx] == 0] = float('nan')
    original_shape = (label_list_org[idx].shape[0], label_list_org[idx].shape[1])
    test_diffs_xgb_reshaped = test_diffs_xgb_patch.reshape(original_shape)
    mae = mean_absolute_error(label_list2[idx], y_pred_xgb_patch)
    rmse = np.sqrt(mean_squared_error(label_list2[idx], y_pred_xgb_patch))
    r2 = r2_score(label_list2[idx], y_pred_xgb_patch)
    print(f"Patch Name: {idx}, MAE: {mae:.3f}, RMSE: {rmse:.3f}, R2: {r2:.3f}")

    year_idx = idx // len(columns)
    col_idx = idx % len(columns)

    if idx != 15:
        cax = axs[year_idx, col_idx].imshow(test_diffs_xgb_reshaped, cmap=cmap, vmin=0, vmax=1000, aspect='equal')
        axs[year_idx, col_idx].axis('off')
    else:
        axs[year_idx, col_idx].axis('off')

plt.subplots_adjust(wspace=0.05, hspace=0.05)
cbar_ax = fig.add_axes([0.15, 0.07, 0.7, 0.01])
cb = fig.colorbar(cax, cax=cbar_ax, orientation='horizontal', label='Ib/900m2')
cbar_ax.tick_params(labelsize=40)
cbar_ax.xaxis.label.set_fontsize(40)
plt.savefig('XGB_all_dry_30_patch.png')
plt.show()

plt.figure(dpi=300)
plt.scatter(mean_diff_list2, y_pred_xgb_patch_list)
plt.show()
