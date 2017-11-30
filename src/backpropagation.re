open Matrix;
open Layer;
open Network;
open Utils;

let copy_weights = (network) => {
    Array.init(
        Array.length(network),
        (i) => {
            Array.make_matrix(
            rows(network[i].weights),
            cols(network[i].weights),
            0.0
        )
        }
    );
};

let copy_biases = (network) => {
    Array.init(
        Array.length(network),
        (i) => Array.make(Array.length(network[i].biases), 0.0)
    );
};

let std_dev = (network, inputs, outputs) => {
    let results = Array.make(Array.length(inputs), Array.make(Array.length(inputs[0]), 0.0));
    for (i in 0 to Array.length(inputs) - 1) {
        let (_, res) = execute_network(network, inputs[i]);
        results[i] = res[Array.length(res) - 1]
    };
    print_string("Expected / Actual: \n");
    List.iter2(
        (a, o) => {
            print_string("    ");
            print_float(o[0]);
            print_string("    |    ");
            print_float(a[0]);
            print_newline();
        },
        Array.to_list(results),
        Array.to_list(outputs)
    );
    let vec_std_dev = Utils.vectorize2_both(Utils.std_deviation);
    let std_devs = vec_std_dev(outputs, results);
    let avg_std_dev = Utils.array_average(std_devs);
    avg_std_dev
};

let backpropagation = (network, input, output) => {
    let net_length = Array.length(network);
    let updated_weights = copy_weights(network);
    let updated_biases = copy_biases(network);
    let (zs, acs) = execute_network(network, input);
    let vec_diff_loss = vectorize2_both(diff_loss);
    let vec_diff_act = vectorize1(diff_relu);
    
    let deltas = Array.init(
        net_length,
        (i) => Array.make(Matrix.rows(network[i].weights), 0.0)
    );
    
    deltas[net_length - 1] = adamard(
        vec_diff_loss(acs[net_length - 1], output),
        vec_diff_act(zs[net_length - 1])
    );

    for (n in net_length - 2 downto 0) {
        let first = Matrix.mult(
            transpose(network[n + 1].weights),
            transpose([|deltas[n + 1]|])
        );
        let second = vec_diff_act(zs[n]);
        let ada = adamard(transpose(first)[0], second);
        deltas[n] = ada; 
    };

    for (l in net_length - 1 downto 1) {
        for (j in 0 to rows(network[l].weights) - 1) {
            for (k in 0 to cols(network[l].weights) - 1) {
                updated_weights[l][j][k] = acs[l - 1][k] *. deltas[l][j];
            }
        };
        for (i in 0 to Array.length(network[l].biases) - 1) {
            updated_biases[l][i] = deltas[l][i]
        };
    };

    (updated_weights, updated_biases)
};

let rec train = (network, learning_rate, inputs, outputs, epochs): network => {
    let sd = std_dev(network, inputs, outputs);
    let batch_size = float_of_int(Array.length(inputs));
    Printf.printf("Std. dev: %.2f \n\n", sd);
    if (epochs <= 0) {
        network
    } else {
        let diff_weights = copy_weights(network);
        let diff_biases = copy_biases(network);
        for (i in 0 to Array.length(inputs) - 1) {
            let (w, b) = backpropagation(network, inputs[i], outputs[i]);
            for (l in 0 to Array.length(diff_weights) - 1) {
                for (j in 0 to rows(diff_weights[l]) - 1) {
                    for (k in 0 to cols(diff_weights[l]) - 1) {
                        diff_weights[l][j][k] = diff_weights[l][j][k] +. w[l][j][k]
                    }
                };
                for (j in 0 to Array.length(diff_biases[l]) - 1) {
                    diff_biases[l][j] = diff_biases[l][j] +. b[l][j]
                };
            };
        };
        for (l in 1 to Array.length(network) - 1) {
            for (j in 0 to rows(diff_weights[l]) - 1) {
                for (k in 0 to cols(diff_weights[l]) - 1) {
                    diff_weights[l][j][k] = network[l].weights[j][k] +. (learning_rate /. batch_size) *. diff_weights[l][j][k]
                }
            };
            for (i in 0 to Array.length(diff_biases[l]) - 1) {
                diff_biases[l][i] = network[l].biases[i] +. (learning_rate /. batch_size) *. diff_biases[l][i]
            }
        };
        let updated_network = change_network(network, diff_weights, diff_biases);
        train(updated_network, learning_rate, inputs, outputs, epochs - 1)
    }
};

let network = build_network(3, (1, 3), 1);

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
    [|1.0|],
    [|0.0|],
    [|1.0|],
    [|0.0|],
    [|1.0|],
    [|0.0|],
    [|1.0|],
    [|0.0|]
|];

/* let network = build_network(3, (2, 3), 3);

let inputs = [|
    [|0.0, 0.0, 0.0|],
    [|0.0, 0.0, 1.0|],
    [|0.0, 1.0, 0.0|],
    [|0.0, 1.0, 1.0|],
    [|1.0, 0.0, 0.0|],
    [|1.0, 0.0, 1.0|],
    [|1.0, 1.0, 0.0|],
|];

let outputs = [|
    [|0.0, 0.0, 1.0|],
    [|0.0, 1.0, 0.0|],
    [|0.0, 1.0, 1.0|],
    [|1.0, 0.0, 0.0|],
    [|1.0, 0.0, 1.0|],
    [|1.0, 1.0, 0.0|],
    [|1.0, 1.0, 1.0|],
|]; */

/* let network = build_network(2, (1, 2), 1);

let inputs = [|
    [|0.0, 0.0|],
    [|0.0, 1.0|],
    [|1.0, 0.0|],
    [|1.0, 1.0|]
|];

let outputs = [|
    [|0.0|],
    [|1.0|],
    [|1.0|],
    [|1.0|]
|]; */

train(network, 0.1, inputs, outputs, 5000);