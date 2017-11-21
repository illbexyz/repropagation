open Matrix;
open Layer;
open Network;

let revert = (a: array('a)) => {
    let length = Array.length(a) - 1;
    let rev_a = Array.make(length + 1, a[0]);
    for (i in length downto 0) {
        rev_a[length - i] = a[i]
    };
    rev_a
};

let copy_weights = (network) => {
    Array.init(
        Array.length(network) - 1,
        (i) => Array.make_matrix(
            rows(network[i + 1].weights),
            cols(network[i + 1].weights),
            0.0
        )
    );
};

let update_weights = (network, new_weights, learning_rate) => {
    network
};

let backpropagation = (network, input, output): array(matrix(float)) => {
    let updated_weights = copy_weights(network);
    for (n in Array.length(network) - 1 downto 1) {
        let curr_layer = network[n];
        let changed_layer = copy_layer(curr_layer);
        for (i in 0 to rows(curr_layer.weights) - 1) {
            for (j in 0 to cols(curr_layer.weights) - 1) {
                changed_layer.weights[i][j] = 2.0;
            }
        };
        updated_weights[n] = changed_layer.weights
    };
    updated_weights
};

let rec train = (network, learning_rate, inputs, outputs, epochs): network => {
    if (epochs <= 0) {
        network
    } else {
        /* let updated_weights
        for (i in 0 to Array.length(inputs) - 1) {

        } */
        let updated_weights = backpropagation(network, inputs, outputs);
        let updated_network = change_network_weights(network, updated_weights);
        train(updated_network, learning_rate, inputs, outputs, epochs - 1)
    }
};

let network = build_network(3, (2, 3), 1);

let inputs = [|
    [|0.0, 0.0, 0.0|],
    [|0.0, 0.0, 1.0|],
    [|0.0, 1.0, 0.0|],
    [|0.0, 1.0, 1.0|],
    [|1.0, 0.0, 0.0|],
    [|1.0, 0.0, 1.0|],
    [|1.0, 1.0, 0.0|],
    [|1.0, 1.0, 1.0|],
|];

let outputs = [|
    1.0,
    0.0,
    1.0,
    0.0,
    1.0,
    0.0,
    1.0,
    0.0
|];

train(network, 0.05, inputs, outputs, 10);