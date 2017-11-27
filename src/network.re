open Layer;

type network = array(layer);

let build_hidden_layers = (n_hidden, n_neurons) => {
    Array.init(
        n_hidden,
        (_) => new_layer(n_neurons, n_neurons, Utils.logistic, Hidden)
    )
};

let build_network = (n_input_neurons, (n_h_layers, n_neurons_hidden), n_output_neurons): network => {
    let input_layer = new_layer(n_input_neurons, 1, Utils.identity, Input);
    let hidden_layers = build_hidden_layers(n_h_layers, n_neurons_hidden);
    let output_layer = new_layer(n_output_neurons, n_neurons_hidden, Utils.logistic, Output);
    Array.concat([
        [|input_layer|],
        hidden_layers,
        [|output_layer|]
    ])
};

let change_network_weights = (network, weights, biases): network => {
    /* Input layer's weights doesn't need to be changed */
    let input_layer = network[0];
    let output_layer_index = Array.length(network) - 1;
    let output_layer = change_layer(
        network[output_layer_index],
        weights[output_layer_index],
        biases[output_layer_index]
    );
    /* The hidden layers are (network.length - 2), from 1 to network.length - 1 */
    let hidden_layers = Array.init(
        Array.length(network) - 2,
        (i) => change_layer(
            network[i + 1],
            weights[i + 1],
            biases[i + 1]
        )
    );
    Array.concat([
        [|input_layer|],
        hidden_layers,
        [|output_layer|]
    ])
};

let execute_network = (network: network, input: array(float)) => {
    let zs = Array.init(
        Array.length(network),
        (i) => Array.make(Matrix.rows(network[i].weights), 0.0)
    );
    let activations = Array.init(
        Array.length(network),
        (i) => Array.make(Matrix.rows(network[i].weights), 0.0)
    );
    let (a, b) = layer_output(network[0], input);
    zs[0] = a;
    activations[0] = b;
    for (i in 1 to Array.length(network) - 1) {
        let (a, b) = layer_output(network[i], activations[i - 1]);
        zs[i] = a;
        activations[i] = b
    };
    (zs, activations)
};