open Jest;
open Expect;
open Matrix;

let oneElm = [|
    [| 0.0 |]
|];

let m10 = [|
    [| 0.0, 0.0, 0.0 |],
    [| 0.0, 0.0, 0.0 |],
    [| 0.0, 0.0, 0.0 |],
|];

let m11 = [|
    [| 1.0, 1.0, 1.0 |],
    [| 1.0, 1.0, 1.0 |],
    [| 1.0, 1.0, 1.0 |],
|];

let m20 = [|
    [| 0.0, 0.0, 0.0 |],
    [| 0.0, 0.0, 0.0 |],
    [| 0.0, 0.0, 0.0 |],
|];

let compl1 = [|
    [| 0.0, 1.1, 0.0 |],
    [| 1.1, 0.0, 1.1 |],
    [| 0.0, 1.1, 0.0 |],
|];
let compl2 = [|
    [| 1.1, 0.0, 1.1 |],
    [| 0.0, 1.1, 0.0 |],
    [| 1.1, 0.0, 1.1 |],
|];
let compl_res = [|
    [| 1.1, 1.1, 1.1 |],
    [| 1.1, 1.1, 1.1 |],
    [| 1.1, 1.1, 1.1 |],
|];

describe("rows", () => {
    test("Works on single element", () =>
        expect(rows(oneElm)) |> toBe(1)
    );
    test("Exact number of rows", () =>
        expect(rows(m10)) |> toBe(3)
    );
});

describe("cols", () => {
    test("Works on single element", () =>
        expect(cols(oneElm)) |> toBe(1)
    );
    test("Exact number of cols", () =>
        expect(cols(m10)) |> toBe(3)
    );
});

describe("sum", () => {
    test("doesn't sum anything on zeroes", () =>
        expect(sum(m10, m20)) |> toEqual(m10)
    );
    test("works on complementary matrixes", () =>
        expect(sum(compl1, compl2)) |> toEqual(compl_res)
    );
});

describe("sub", () => {
    test("doesn't sub anything on zeroes", () =>
        expect(sub(m10, m20)) |> toEqual(m10)
    );
    test("works on complementary matrixes", () =>
        expect(sub(compl_res, compl2)) |> toEqual(compl1)
    );
});

describe("mult", () => {
    test("doesn't do anything on zeroes", () =>
        expect(mult(m10, m20)) |> toEqual(m10)
    );
    let a = [| [|1.0|], [|1.0|], [|1.0|], [|1.0|] |];
    let b = [| [|2.0, 2.0, 2.0|] |];
    let res = mult(a, b);
    let expected_rows = rows(a);
    let expected_cols = cols(b);
    test("the result has the exact number of rows", () => {
        expect(rows(res)) |> toEqual(expected_rows);
    });
    test("the result has the exact number of columns", () => {
        expect(cols(res)) |> toEqual(expected_cols);
    });
    let compl_res = [|
        [|
            (compl1[0][0] *. compl2[0][0]) +.
            (compl1[0][1] *. compl2[1][0]) +.
            (compl1[0][2] *. compl2[2][0]),

            (compl1[0][0] *. compl2[0][1]) +.
            (compl1[0][1] *. compl2[1][1]) +.
            (compl1[0][2] *. compl2[2][1]),

            (compl1[0][0] *. compl2[0][2]) +.
            (compl1[0][1] *. compl2[1][2]) +.
            (compl1[0][2] *. compl2[2][2])
        |], [|
            (compl1[1][0] *. compl2[0][0]) +.
            (compl1[1][1] *. compl2[1][0]) +.
            (compl1[1][2] *. compl2[2][0]),

            (compl1[1][0] *. compl2[0][1]) +.
            (compl1[1][1] *. compl2[1][1]) +.
            (compl1[1][2] *. compl2[2][1]),

            (compl1[1][0] *. compl2[0][2]) +.
            (compl1[1][1] *. compl2[1][2]) +.
            (compl1[1][2] *. compl2[2][2])
        |], [|
            (compl1[2][0] *. compl2[0][0]) +.
            (compl1[2][1] *. compl2[1][0]) +.
            (compl1[2][2] *. compl2[2][0]),

            (compl1[2][0] *. compl2[0][1]) +.
            (compl1[2][1] *. compl2[1][1]) +.
            (compl1[2][2] *. compl2[2][1]),

            (compl1[2][0] *. compl2[0][2]) +.
            (compl1[2][1] *. compl2[1][2]) +.
            (compl1[2][2] *. compl2[2][2])
        |]
    |];
    test("does the correct operation", () =>
        expect(mult(compl1, compl2)) |> toEqual(compl_res)
    );
});

describe("apply", () => {
    test("applies the function to every element", () => {
        let m = m10;
        let expected = sum(m, m11);
        let res = apply(m10, (e) => e +. 1.0);
        expect(res) |> toEqual(expected)
    });
});

describe("transpose", () => {
    let a = [| [|1.0|], [|1.0|], [|1.0|], [|1.0|] |];
    let a_transposed = transpose(a); 
    test("the result has the exact number of rows", () => {
        let expected = cols(a);
        let actual = rows(a_transposed);
        expect(actual) |> toBe(expected)
    });
    test("the result has the exact number of columns", () => {
        let expected = rows(a);
        let actual = cols(a_transposed);
        expect(actual) |> toBe(expected)
    });
    test("applying transpose two times we get the first matrix", () => {
        let expected = a;
        let actual = transpose(a_transposed);
        expect(actual) |> toEqual(expected)
    });
});