import mlflow
import mlflow.sklearn
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import plot_roc_curve
import matplotlib.pyplot as plt
import seaborn as sns


def train_model(model, X_train, y_train):
    """
    Function to train the model and log metrics.
    Args:
        - model: Model Object which is to be trained.
        - X_train: Independent features to use to train.
        - y_train: Dependent feature to use to train.
    """
    # fitting the model
    model = model.fit(X_train, y_train)

    # training accuracy
    train_accuracy = model.score(X_train, y_train)

    # logging the metric in MLFlow
    mlflow.log_metric('train_accuracy', train_accuracy)
    print(f"Training accuracy: {train_accuracy:.2%}")


def evaluate_model(model, X_test, y_test):
    """
    Function to Evaluate the performance of the Model.
    Args:
        - model: Trained Model Object to evaluate
        - X_test, y_test: Test data which is to be used to evaluate the model.
    """
    # predicting
    y_preds = model.predict(X_test)

    # metrics:
    eval_accuracy = model.score(X_test, y_test)
    auc_score = roc_auc_score(y_test, y_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_preds, average='binary')

    # logging metrics:
    mlflow.log_metric('Validation_Accuracy', eval_accuracy)
    mlflow.log_metric('AUC_Score', auc_score)
    mlflow.log_metric('Precision', precision)
    mlflow.log_metric('Recall', recall)
    mlflow.log_metric('F1_score', f1)

    # printing the metrics:
    print("Validation Accuracy:", eval_accuracy)
    print("AUC Score:", auc_score)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    # plotting the ROC curve:
    roc_curve = plot_roc_curve(model, X_test, y_test, name='ROC Curve')
    plt.savefig("artifacts/roc_plot.png")
    plt.show()
    plt.clf()

    # plotting the confusion matrix:
    conf_matrix = confusion_matrix(y_test, y_preds)
    ax = sns.heatmap(conf_matrix, annot=True, fmt='g')
    ax.invert_xaxis()
    ax.invert_yaxis()
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig('artifacts/confusion_matrix.png')

    # logging these figures:
    mlflow.log_artifact("artifacts/roc_plot.png")
    mlflow.log_artifact("artifacts/confusion_matrix.png")
