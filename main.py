import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def one_hot_encode(output):
    max_index = np.argmax(output)
    num_classes = output.shape[1]
    one_hot_result = np.zeros(num_classes)
    one_hot_result[max_index] = 1
    return one_hot_result


def classify_result(output):
    one_hot_result = one_hot_encode(output)
    if np.array_equal(one_hot_result, [1, 0, 0]):
        return "Normal hyperthyroid"
    elif np.array_equal(one_hot_result, [0, 1, 0]):
        return "Hyper function"
    elif np.array_equal(one_hot_result, [0, 0, 1]):
        return "Subnormal functioning"

    return "Unknown"


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    sigmoid_x = sigmoid(x)
    return sigmoid_x * (1 - sigmoid_x)


def calculate_loss(output, target):
    return np.mean(np.abs(output - target))  # MSE


def feedforward(training_vec, l1_weights, l1_bias, l2_weights, l2_bias):
    hidden_layer_input = np.add(l1_bias, np.dot(training_vec, l1_weights))
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.add(l2_bias, np.dot(hidden_layer_output, l2_weights))
    output_layer_output = sigmoid(output_layer_input)

    return hidden_layer_output, output_layer_output


def train(df_train_data, df_train_target, df_val_data, df_val_target):
    epochs = 400
    learning_rate = 0.2
    consecutive_correct_threshold = 0
    training_losses = []
    val_accuracies = []

    # Weights between input and hidden layer
    l1_weights = np.random.normal(0, 1, size=(21, 10))
    # biases between input and hidden layer
    l1_bias = np.random.normal(0, 1, size=(1, 10))
    # Weights between hidden layer and output
    l2_weights = np.random.normal(0, 1, size=(10, 3))
    # biases between hidden layer and output
    l2_bias = np.random.normal(0, 1, size=(1, 3))

    for _ in range(epochs):
        epoch_loss = 0

        for row_index in range(df_train_data.shape[0]):
            train_vec = df_train_data.iloc[row_index].to_numpy()
            train_target = df_train_target.iloc[row_index].to_numpy()

            # Feedforward
            hidden_layer_output, final_output = feedforward(train_vec, l1_weights, l1_bias, l2_weights, l2_bias)

            # Backpropagation
            delta_output = sigmoid_derivative(final_output) * (train_target - final_output)
            delta_hidden = sigmoid_derivative(hidden_layer_output) * np.dot(delta_output, l2_weights.T)

            # Weight and bias updates
            l2_weights += learning_rate * np.outer(hidden_layer_output, delta_output)
            l2_bias += learning_rate * delta_output

            l1_weights += learning_rate * np.outer(train_vec, delta_hidden)
            l1_bias += learning_rate * delta_hidden

            # Calculate Loss
            loss = calculate_loss(final_output, train_target)
            epoch_loss += loss
        training_losses.append(epoch_loss)

        # Calculate accuracy after each epoch to plot it at the end
        val_accuracy = calculate_accuracy(df_val_data, df_val_target, l1_weights, l1_bias, l2_weights, l2_bias)
        val_accuracies.append(val_accuracy)

        # Validation
        for row_index in range(df_val_data.shape[0]):
            val_vec = df_val_data.iloc[row_index].to_numpy()
            val_target = df_val_target.iloc[row_index].to_numpy()

            _, val_output = feedforward(val_vec, l1_weights, l1_bias, l2_weights, l2_bias)
            val_result = one_hot_encode(val_output)

            if np.array_equal(val_result, val_target):
                consecutive_correct_threshold += 1
            else:
                consecutive_correct_threshold = 0

        if consecutive_correct_threshold == 6:
            print("training finished by reaching the maximum set value for consecutive correct validation")
            break

    return l1_weights, l1_bias, l2_weights, l2_bias, training_losses, val_accuracies


def test(input_data, l1_weights, l1_bias, l2_weights, l2_bias):
    _, final_output = feedforward(input_data, l1_weights, l1_bias, l2_weights, l2_bias)
    return final_output


def calculate_accuracy(df_test_data, df_test_target, l1_weights, l1_bias, l2_weights, l2_bias):
    num_testing_data = df_test_data.shape[0]
    num_correct_answers = 0

    for row_index in range(num_testing_data):
        test_vec = df_test_data.iloc[row_index].to_numpy()
        test_target = df_test_target.iloc[row_index].to_numpy()

        _, test_output = feedforward(test_vec, l1_weights, l1_bias, l2_weights, l2_bias)
        test_result = one_hot_encode(test_output)

        if np.array_equal(test_result, test_target):
            num_correct_answers += 1

    accuracy = round((num_correct_answers / num_testing_data) * 100, 2)

    return accuracy


def plot_training_history(training_losses, accuracies):
    epochs = len(training_losses)

    # Plot Training Loss
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), training_losses, label='Training Loss', color='blue')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), accuracies, label='Accuracy', color='green')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()


def main():
    VAL_SPLIT = .1
    TEST_SPLIT = .2

    input_data = pd.read_excel("./dataset/thyroidInputs.xlsx", names=range(7200), header=None).transpose()
    target_data = pd.read_excel("./dataset/thyroidTargets.xlsx", names=range(7200), header=None).transpose()

    index_vector = np.linspace(0, 7199, 7200, dtype=np.int16)
    np.random.shuffle(index_vector)

    df_test_data = input_data.iloc[index_vector[:int(TEST_SPLIT*input_data.shape[0])]]
    df_val_data = input_data.iloc[index_vector[int(TEST_SPLIT*input_data.shape[0]): int((TEST_SPLIT + VAL_SPLIT)*input_data.shape[0])]]
    df_train_data = input_data.iloc[index_vector[int((TEST_SPLIT+VAL_SPLIT)*input_data.shape[0]):]]

    df_test_target = target_data.iloc[index_vector[:int(TEST_SPLIT*target_data.shape[0])]]
    df_val_target = target_data.iloc[index_vector[int(TEST_SPLIT*target_data.shape[0]): int((TEST_SPLIT + VAL_SPLIT)*target_data.shape[0])]]
    df_train_target = target_data.iloc[index_vector[int((TEST_SPLIT+VAL_SPLIT)*target_data.shape[0]):]]

    print("start training")
    l1_weights, l1_bias, l2_weights, l2_bias, training_losses, val_accuracies = train(df_train_data, df_train_target, df_val_data, df_val_target)
    print("finish training \n")

    print("Calculating accuracy... \n")
    accuracy = calculate_accuracy(df_test_data, df_test_target, l1_weights, l1_bias, l2_weights, l2_bias)
    print(f"Accuracy of the model is {accuracy}%\n")

    plot_training_history(training_losses, val_accuracies)

    # test_input = [0.38, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0.0039, 0.022, 0.061, 0.076, 0.08]  # result of this input = [0,0,1]
    # output = test(test_input, l1_weights, l1_bias, l2_weights, l2_bias)
    # print("output: ", output)

    # result = classify_result(output)
    # print(result)


if __name__ == "__main__":
    main()
