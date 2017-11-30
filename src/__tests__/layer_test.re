open Jest;
open Expect;
open Layer;

describe("randomize_weights", () => {
    let rows = 3;
    let cols = 4;
    let weights = randomize_weights(rows, cols);
    test("has the correct number of rows", () => {
        let expected = rows;
        let actual = Matrix.rows(weights);
        expect(actual) |> toBe(expected)
    });
    test("has the correct number of columns", () => {
        let expected = cols;
        let actual = Matrix.cols(weights);
        expect(actual) |> toBe(expected)
    });
});

describe("execute", () => {
    test("input layer with 1.0 weights and 0.0 biases returns the input", () => {
        let n_layer = new_layer(
            ~n_neurons=3,
            ~n_inputs=1,
            ~activation=Utils.identity,
            ~l_type=Input
        );
        let weights = [|
            [| 1.0 |],
            [| 1.0 |],
            [| 1.0 |],
        |];
        let biases = [| 0.0, 0.0, 0.0 |];
        let layer = change_layer(n_layer, weights, biases);
        let input = [| 0.5, 0.5, 0.5 |];
        let expected = input;
        let (_, actual) = execute(layer, input);
        expect(actual) |> toEqual(expected);
    });
    test("hidden layer with 1.0 weights and 0.0 biases returns the input multiplied by his input number", () => {
        let n_layer = new_layer(
            ~n_neurons=3,
            ~n_inputs=3,
            ~activation=Utils.identity,
            ~l_type=Hidden
        );
        let weights = [|
            [| 1.0, 1.0, 1.0 |],
            [| 1.0, 1.0, 1.0 |],
            [| 1.0, 1.0, 1.0 |],
        |];
        let biases = [| 0.0, 0.0, 0.0 |];
        let layer = change_layer(n_layer, weights, biases);
        let input = [| 0.5, 0.5, 0.5 |];
        let expected = Array.make(3, input[0] *. 3.0);
        let (_, actual) = execute(layer, input);
        expect(actual) |> toEqual(expected);
    });
});