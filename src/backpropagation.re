open Matrix;
open Layer;
open Network;
open Utils;

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
    Js.log("res");
    Js.log(results);
    Js.log("out");
    Js.log(outputs);
    let std_dev = Utils.std_deviation(
        ~expecteds=outputs,
        ~actuals=results[Array.length(results) - 1]
    );
    std_dev
};

let backpropagation = (network, input, output) => {
    /* Js.log(network); Js.log(); */
    let net_length = Array.length(network);
    let updated_weights = copy_weights(network);
    let updated_biases = copy_biases(network);
    let (zs, acs) = execute_network(network, input);
    let vec_diff_loss = vectorize2(diff_loss);
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
        let first = Matrix.dot(
            transpose(network[n + 1].weights),
            [|deltas[n + 1]|]
        );
        let second = vec_diff_act(zs[n]);
        /* Js.log("first"); Js.log(first);
        Js.log("second"); Js.log(second); */
        deltas[n] = adamard(transpose(first)[0], second);
        /* Js.log("delta"); Js.log(deltas[n]); */
    };

    /* Js.log(updated_weights); */
    /* Printf.printf("rows: %i\n", rows(updated_weights[2]));
    Printf.printf("cols: %i\n", cols(updated_weights[2]));
    Printf.printf("rows: %i\n", rows(network[2].weights));
    Printf.printf("cols: %i\n", cols(network[2].weights)); */
    for (l in net_length - 1 downto 1) {
        for (j in 0 to rows(network[l].weights) - 1) {
            for (k in 0 to cols(network[l].weights) - 1) {
                /* Printf.printf("%i %i %i\n", l, j, k); */
                /* Js.log(acs); print_newline();
                Js.log(deltas); print_newline(); */
                updated_weights[l][j][k] = acs[l - 1][k] *. deltas[l][j];
            }
        };
        for (i in 0 to Array.length(network[l].biases) - 1) {
            updated_biases[l][i] = deltas[l][i]
        };
    };

    /* Js.log(updated_weights); */
    (updated_weights, updated_biases)
};

let rec train = (network, learning_rate, inputs, outputs, epochs): network => {
    let sd = std_dev(network, inputs, outputs);
    Printf.printf("Std. dev.: %.2f \n", sd);
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
        let avg_weights = Array.map(
            (weights) => Matrix.apply(weights, (x) => x /. float_of_int(Array.length(inputs))),
            diff_weights
        );
        let avg_biases = Array.map(
            (biases) => vectorize2((x, y) => x /. y)(biases)(float_of_int(Array.length(inputs))),
            diff_biases
        );
        for (l in 1 to Array.length(network) - 1) {
            for (j in 0 to rows(avg_weights[l]) - 1) {
                for (k in 0 to cols(avg_weights[l]) - 1) {
                    avg_weights[l][j][k] = network[l].weights[j][k] -. (learning_rate *. avg_weights[l][j][k])
                }
            };
            for (i in 0 to Array.length(avg_biases[l]) - 1) {
                avg_biases[l][i] = network[l].biases[i] -. (learning_rate *. avg_biases[l][i])
            }
        };
        let updated_network = change_network_weights(network, avg_weights, avg_biases);
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

/* backpropagation(network, inputs[0], outputs[0]); */

train(network, 0.3, inputs, outputs, 100000000);
/* let result = execute_network(network, inputs[1]); */

/* Js.log(result); */
/* let results = Array.make(Array.length(inputs), Array.make(Array.length(inputs[0]), 0.0));
for (i in 0 to Array.length(inputs) - 1) {
    let outputs = execute_network(network, inputs[i]);
    results[i] = outputs[Array.length(outputs) - 1]
};
Js.log(results);
let std_dev = Utils.std_deviation(
    ~expecteds=outputs,
    ~actuals=results[Array.length(results) - 1]
);

Printf.printf("Std. dev.: %.2f", std_dev);
print_newline(); */