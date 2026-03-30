#!/usr/bin/env node
"use strict";

const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, HeadingLevel, BorderStyle, WidthType, ShadingType,
  LevelFormat, PageNumber, Footer, TabStopType, TabStopPosition,
} = require("docx");

// ── read JSON from first CLI arg ─────────────────────────────────────────────
const data = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const { qoi, task, train_pct, file_name, generated_at, models, recommendation } = data;

// ── colour palette ───────────────────────────────────────────────────────────
const BLUE   = "1A56DB";
const DARK   = "1A1A2E";
const GREY   = "6B7280";
const GREEN  = "15803D";
const GREEN_BG = "DCFCE7";
const STRIPE = "EEF2FF";
const WHITE  = "FFFFFF";

// ── tiny helpers ─────────────────────────────────────────────────────────────
const run = (text, opts = {}) => new TextRun({
  text,
  font: "Calibri",
  size: (opts.size || 11) * 2,
  bold:   opts.bold   || false,
  italics: opts.italic || false,
  color:  opts.color  || DARK,
  break: opts.break || 0,
});

const para = (children, opts = {}) => new Paragraph({
  alignment: opts.align || AlignmentType.LEFT,
  spacing: {
    before: (opts.before || 0) * 20,
    after:  (opts.after  || 6) * 20,
    line: 276,
  },
  children: Array.isArray(children) ? children : [children],
});

const h1 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_1,
  spacing: { before: 280, after: 80 },
  border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: BLUE, space: 1 } },
  children: [new TextRun({ text, font: "Calibri", size: 28, bold: true, color: BLUE })],
});

const h2 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_2,
  spacing: { before: 180, after: 60 },
  children: [new TextRun({ text, font: "Calibri", size: 24, bold: true, color: "374151" })],
});

const kv = (key, value) => new Paragraph({
  spacing: { before: 20, after: 40 },
  children: [
    new TextRun({ text: `${key}: `, font: "Calibri", size: 20, bold: true, color: "374151" }),
    new TextRun({ text: String(value), font: "Calibri", size: 20, color: DARK }),
  ],
});

const bullet = (text) => new Paragraph({
  numbering: { reference: "bullets", level: 0 },
  spacing: { before: 20, after: 40 },
  children: [new TextRun({ text, font: "Calibri", size: 20, color: DARK })],
});

const spacer = () => new Paragraph({ spacing: { before: 0, after: 120 }, children: [new TextRun("")] });

// ── cell builder ─────────────────────────────────────────────────────────────
const border = (color = "CCCCCC") => ({
  top:    { style: BorderStyle.SINGLE, size: 4, color },
  bottom: { style: BorderStyle.SINGLE, size: 4, color },
  left:   { style: BorderStyle.SINGLE, size: 4, color },
  right:  { style: BorderStyle.SINGLE, size: 4, color },
});

const cell = (text, width, opts = {}) => new TableCell({
  width: { size: width, type: WidthType.DXA },
  borders: border(opts.headerCell ? WHITE : "CCCCCC"),
  shading: { fill: opts.fill || WHITE, type: ShadingType.CLEAR },
  margins: { top: 60, bottom: 60, left: 120, right: 120 },
  children: [new Paragraph({
    alignment: opts.center ? AlignmentType.CENTER : AlignmentType.LEFT,
    spacing: { before: 0, after: 0 },
    children: [new TextRun({
      text,
      font: "Calibri",
      size: 20,
      bold: opts.bold || false,
      color: opts.color || DARK,
    })],
  })],
});

// ── model static descriptions ─────────────────────────────────────────────────
const MODEL_DESC = {
  regression: {
    m1: {
      name: "Ordinary Least Squares (OLS) Linear Regression",
      desc: "OLS estimates coefficients by minimising the residual sum of squares between observed and predicted values. The solution is obtained analytically via the normal equations, making it deterministic and parameter-free. It assumes a linear relationship between predictors and response, homoscedastic independent errors, and no perfect multicollinearity.",
      assumptions: [
        "Linearity: E[Y|X] is linear in the predictors.",
        "Independence: error terms are uncorrelated.",
        "Homoscedasticity: constant error variance across fitted values.",
        "No perfect multicollinearity among predictor variables.",
      ],
    },
    m2: {
      name: "Multi-Layer Perceptron Regressor (Neural Network)",
      desc: "A fully-connected feed-forward neural network with one or more hidden layers, each applying a non-linear ReLU activation. Weights are optimised via back-propagation using the Adam solver, minimising mean squared error. L2 regularisation (alpha) penalises large weights to limit overfitting. All features are standardised before training.",
      assumptions: [
        "Sufficient observations relative to the number of network parameters.",
        "Features are standardised to comparable scales (applied in this pipeline).",
        "Convergence within the specified max_iter epochs is not guaranteed for all datasets.",
      ],
    },
    m3: {
      name: "Random Forest Regressor",
      desc: "An ensemble of decision trees, each trained on a bootstrap sample of the data with a random feature subset considered at every split (bagging + feature randomness). Predictions are the mean of all tree outputs. Random Forests are robust to outliers and capture non-linear relationships without requiring feature scaling.",
      assumptions: [
        "Enough trees to stabilise ensemble variance (typically 100 or more).",
        "max_depth controls individual tree complexity; unlimited depth may overfit on small datasets.",
        "Feature scaling is not required.",
      ],
    },
  },
  classification: {
    m1: {
      name: "Binary Logistic Regression (GLM)",
      desc: "A generalised linear model that models the log-odds of the binary outcome as a linear combination of predictors. The sigmoid function maps predictions to probabilities in [0,1]. Coefficients are estimated by maximum likelihood. The regularisation parameter C (inverse of penalty strength) controls the fit-complexity trade-off.",
      assumptions: [
        "Binary outcome variable.",
        "Log-odds are linearly related to the predictors.",
        "Independence of observations.",
        "No severe multicollinearity among predictors.",
      ],
    },
    m2: {
      name: "Multi-Layer Perceptron Classifier (Neural Network)",
      desc: "A feed-forward neural network for binary classification with a logistic output layer. Trained by back-propagation (Adam) minimising cross-entropy loss. L2 regularisation (alpha) and a configurable maximum iteration count help prevent overfitting. Features are standardised before training.",
      assumptions: [
        "Sufficient data for the chosen architecture depth and width.",
        "Features are standardised before training (applied in this pipeline).",
        "Convergence within max_iter epochs is not guaranteed.",
      ],
    },
    m3: {
      name: "Random Forest Classifier",
      desc: "An ensemble of classification trees trained on bootstrap samples with random feature subsets at each split. The final class prediction is by majority vote across all trees. Captures complex non-linear decision boundaries and is robust to irrelevant features without requiring feature scaling.",
      assumptions: [
        "Sufficient trees to reduce ensemble variance.",
        "max_depth limits per-tree complexity to control overfitting.",
        "Feature scaling is not required.",
      ],
    },
  },
};

// ── build document children ──────────────────────────────────────────────────
const children = [];

// Title
children.push(
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 0, after: 80 },
    children: [new TextRun({ text: "ML Workbench — Model Comparison Report", font: "Calibri", size: 44, bold: true, color: DARK })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 0, after: 40 },
    children: [new TextRun({ text: `Generated: ${generated_at}`, font: "Calibri", size: 20, color: GREY })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 0, after: 280 },
    children: [new TextRun({ text: `Source file: ${file_name}`, font: "Calibri", size: 20, color: GREY })],
  }),
);

// ── Section 1: Problem Setup ──────────────────────────────────────────────────
children.push(
  h1("1. Problem Setup"),
  kv("Quantity of Interest (QOI)", qoi),
  kv("Task type", task === "regression" ? "Regression" : "Binary Classification"),
  kv("Training / test split", `${train_pct}% training  /  ${100 - train_pct}% test`),
  kv("Feature encoding", "Categorical predictors one-hot encoded (drop-first); all features standardised (zero mean, unit variance)"),
  kv("Random seed", "42 (fixed for reproducibility)"),
  spacer(),
);

// ── Section 2: Model Descriptions ────────────────────────────────────────────
children.push(h1("2. Model Descriptions & Parameters"));

const modelKeys = ["m1", "m2", "m3"];
modelKeys.forEach(key => {
  const m    = models[key];
  const meta = MODEL_DESC[task][key];
  if (!m) return;

  children.push(h2(meta.name));
  children.push(para(
    [run(meta.desc, { size: 10 })],
    { after: 8 }
  ));

  // Assumptions
  children.push(para([run("Key assumptions:", { bold: true, size: 10 })], { after: 4 }));
  meta.assumptions.forEach(a => children.push(bullet(a)));

  // Parameters
  const params = m.params || {};
  const paramEntries = Object.entries(params);
  if (paramEntries.length > 0) {
    children.push(para([run("Parameters used:", { bold: true, size: 10 })], { before: 8, after: 4 }));
    paramEntries.forEach(([k, v]) => children.push(kv(k, v)));
  } else {
    children.push(para([run("Parameters: Fitted analytically — none required.", { italic: true, size: 10 })], { after: 4 }));
  }
  children.push(spacer());
});

// ── Section 3: Performance Comparison Table ───────────────────────────────────
children.push(h1("3. Performance Comparison"));

const isReg = task === "regression";
children.push(para([run(
  isReg
    ? "The table below reports Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) on training and test sets. Lower values indicate better fit. Test-set performance is the primary indicator of generalisation ability."
    : "The table below reports Accuracy and F1 Score on training and test sets. Higher values indicate better classification performance. Test-set metrics are the primary indicators of generalisation.",
  { size: 10 }
)], { after: 120 }));

// Column config
const cols = isReg
  ? { labels: ["Model", "Train MSE", "Test MSE", "Train RMSE", "Test RMSE"], widths: [2800, 1640, 1640, 1640, 1640] }
  : { labels: ["Model", "Train Acc.", "Test Acc.", "Train F1", "Test F1"],    widths: [2800, 1640, 1640, 1640, 1640] };

// Find best model index
const testVals = modelKeys.map(k => {
  const m = models[k];
  return isReg ? (m ? m.test_mse : Infinity) : (m ? m.test_acc : -Infinity);
});
const bestIdx = isReg
  ? testVals.indexOf(Math.min(...testVals))
  : testVals.indexOf(Math.max(...testVals));

// Header row
const headerRow = new TableRow({
  tableHeader: true,
  children: cols.labels.map((label, ci) =>
    cell(label, cols.widths[ci], { fill: BLUE, color: WHITE, bold: true, center: ci > 0, headerCell: true })
  ),
});

// Data rows
const dataRows = modelKeys.map((key, ri) => {
  const m   = models[key];
  const bg  = ri === bestIdx ? GREEN_BG : (ri % 2 === 0 ? WHITE : STRIPE);
  const isBest = ri === bestIdx;
  const vals = isReg
    ? [m.name, m.train_mse.toFixed(6), m.test_mse.toFixed(6), m.train_rmse.toFixed(6), m.test_rmse.toFixed(6)]
    : [m.name, m.train_acc.toFixed(4), m.test_acc.toFixed(4),  m.train_f1.toFixed(4),   m.test_f1.toFixed(4)];
  return new TableRow({
    children: vals.map((v, ci) =>
      cell(v, cols.widths[ci], { fill: bg, bold: isBest, center: ci > 0, color: isBest && ci > 0 ? GREEN : DARK })
    ),
  });
});

children.push(
  new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: cols.widths,
    rows: [headerRow, ...dataRows],
  }),
  para([run(
    isReg ? "Green row = lowest Test MSE (best generalisation)." : "Green row = highest Test Accuracy (best generalisation).",
    { italic: true, size: 9, color: GREEN }
  )], { before: 6, after: 80 }),
  spacer(),
);

// ── Section 4: Recommendation ────────────────────────────────────────────────
children.push(
  h1("4. Recommendation"),
  para([run(recommendation, { size: 10 })], { after: 80 }),
  spacer(),
);

// ── Section 5: Methodology Notes ─────────────────────────────────────────────
children.push(h1("5. Methodology Notes"));
const notes = [
  "All models trained with random seed 42 for reproducibility across runs.",
  "Features were standardised (zero mean, unit variance) prior to fitting — critical for OLS/Logistic Regression and MLP, which are sensitive to feature scale.",
  "Categorical predictors were one-hot encoded with drop-first to eliminate perfect multicollinearity.",
  "Rows with missing values in any predictor column were excluded before modelling.",
  "A large gap between training and test metrics suggests overfitting; consider stronger regularisation or simpler model architecture.",
  "For the MLP, a ConvergenceWarning from sklearn indicates max_iter may be too low — increase it if this occurs.",
  "Random Forest test performance stabilises as n_estimators increases; a minimum of 100 trees is recommended.",
  "Confusion matrices (classification) or residual plots (regression) should be examined alongside summary metrics for a fuller picture.",
];
notes.forEach(n => children.push(bullet(n)));

// ── Assemble document ─────────────────────────────────────────────────────────
const doc = new Document({
  numbering: {
    config: [{
      reference: "bullets",
      levels: [{
        level: 0,
        format: LevelFormat.BULLET,
        text: "\u2022",
        alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } },
      }],
    }],
  },
  styles: {
    default: {
      document: { run: { font: "Calibri", size: 22 } },
    },
    paragraphStyles: [
      {
        id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run:       { size: 28, bold: true, font: "Calibri", color: BLUE },
        paragraph: { spacing: { before: 280, after: 80 }, outlineLevel: 0 },
      },
      {
        id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run:       { size: 24, bold: true, font: "Calibri", color: "374151" },
        paragraph: { spacing: { before: 180, after: 60 }, outlineLevel: 1 },
      },
    ],
  },
  sections: [{
    properties: {
      page: {
        size:   { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
      },
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [
            new TextRun({ text: "ML Workbench Report  |  Page ", font: "Calibri", size: 18, color: GREY }),
            new TextRun({ children: [PageNumber.CURRENT], font: "Calibri", size: 18, color: GREY }),
          ],
        })],
      }),
    },
    children,
  }],
});

Packer.toBuffer(doc).then(buf => {
  process.stdout.write(buf);
});
