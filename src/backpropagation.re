open Layer;

type network = array(layer);

let how_many_inputs = 3;
let how_many_neurons = 3;

let e = 2.7182818284590452353602874713527;

let revert = (a: array('a)) => {
    let length = Array.length(a) - 1;
    let rev_a = Array.make(length + 1, a[0]);
    for (i in length downto 0) {
        rev_a[length - i] = a[i]
    };
    rev_a
};

let identity = (x) => x;

let diff_tanh = (x) => {
    1.0 -. tanh(x) *. tanh(x)
};

let diff_loss = (expected, output) => {
    (output -. expected)
};

let build_hidden_layers = (n_hidden) => {
    Array.init(n_hidden, (_) => new_layer(how_many_neurons, how_many_inputs, tanh, Hidden))
};

let build_network = (n_hidden): network => {
    let input_layer = new_layer(how_many_neurons, 1, identity, Input);
    let hidden_layers = build_hidden_layers(n_hidden);
    let output_layer = new_layer(1, how_many_inputs, tanh, Output);
    Array.concat([
        [|input_layer|],
        hidden_layers,
        [|output_layer|]
    ])
};

let execute_network = (network: network, inputs) => {
    Array.fold_left((a, b) => layer_output(b, a), inputs, network)
};

let rec backpropagate = (network, outputs, index, (i, j), diff_act) => {
    let layer = network[index];
    let out = outputs[index][0];
    switch (layer.l_type) {
    | Input => Array.fold_left((a, b) => a +. b, 0.0, out)
    | _ =>
        Array.fold_left(
            (a, b) =>
                a +. (
                    layer.weights[i][index == 0 ? 0 : j]
                    *. diff_act(b)
                    *. backpropagate(network, outputs, index + 1, (i, j), diff_act)
                ),
            0.0,
            out
        )
    }
};

let backpropagation = (network, inputs, (i, j), expected) => {
    let outputs = Array.make(Array.length(network), [|[||]|]);
    for (i in 0 to Array.length(network) - 1) {
        outputs[i] = layer_output(network[i], i == 0 ? inputs : outputs[i - 1])
    };
    let rev_net = revert(network);
    let rev_outs = revert(outputs);
    let last_out = outputs[Array.length(outputs) - 1][0][0];
    (
        diff_loss(expected, last_out)
        *. diff_tanh(last_out)
        *. backpropagate(rev_net, rev_outs, 0, (i, j), diff_tanh)
    )
};

let train = (network, learning_rate, input, output) => {
    let d_weights = Array.make_matrix(3, 3, 0.0);
    for (i in 0 to 2) {
        for (j in 0 to 2) {
            d_weights[i][j] = backpropagation(network, input, (i, j), output)
        }
    };
    
};

let network = build_network(2);

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
    0.0,
    1.0,
    2.0,
    3.0,
    4.0,
    5.0,
    6.0,
    7.0
|];

Js.log(backpropagation(network, [|inputs[1]|], (0, 0), outputs[1]));
Js.log(backpropagation(network, [|inputs[1]|], (0, 1), outputs[1]));
Js.log(backpropagation(network, [|inputs[1]|], (0, 2), outputs[1]));
Js.log(backpropagation(network, [|inputs[1]|], (1, 0), outputs[1]));
Js.log(backpropagation(network, [|inputs[1]|], (1, 1), outputs[1]));
Js.log(backpropagation(network, [|inputs[1]|], (1, 2), outputs[1]));