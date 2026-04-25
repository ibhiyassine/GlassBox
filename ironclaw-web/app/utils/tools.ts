/**
 * LLM Function-Calling schemas for the GlassBox IronClaw toolkit.
 *
 * Each declaration mirrors one Python function in `glassbox.ironclaw.api`
 * and is formatted for the Gemini `functionDeclarations` spec.
 */

import type { FunctionDeclaration } from '@google/genai'

/* ──────────────────────────── Inspect ──────────────────────────── */
export const inspectDataTool: FunctionDeclaration = {
    name: 'inspect_data',
    description:
        'Load a CSV file already present in the WASM filesystem and run a full exploratory data analysis (EDA). ' +
        'This MUST be called first before any cleaning or training. ' +
        'Returns column types, missing-value counts, basic statistics, and distribution summaries.',
    parameters: {
        type: 'object' as any,
        properties: {
            csv_filename: {
                type: 'string' as any,
                description:
                    'The name of the CSV file to inspect (e.g. "data.csv"). The file is already loaded into the virtual filesystem.',
            },
        },
        required: ['csv_filename'],
    },
}

/* ─────────────────────────── Clean ─────────────────────────── */
export const cleanDataTool: FunctionDeclaration = {
    name: 'clean_data',
    description:
        'Apply data-cleaning transformations to the dataset currently held in memory. ' +
        'The dataset must have been loaded by inspect_data first. ' +
        'Configure imputation, outlier handling, encoding, and scaling strategies.',
    parameters: {
        type: 'object' as any,
        properties: {
            imputation_strategy: {
                type: 'object' as any,
                description:
                    'Mapping of column names to imputation methods. Supported methods: "mean", "median", "mode", "constant".',
            },
            outlier_handling: {
                type: 'object' as any,
                description:
                    'Mapping of column names to outlier handling methods. Supported methods: "clip", "drop", "none".',
            },
            encoding_strategy: {
                type: 'object' as any,
                description:
                    'Mapping of column names to encoding methods. Supported methods: "onehot", "label", "none".',
            },
            scale: {
                type: 'string' as any,
                description:
                    'Global scaling method to apply. One of: "standard", "minmax", "none".',
            },
        },
        required: [
            'imputation_strategy',
            'outlier_handling',
            'encoding_strategy',
            'scale',
        ],
    },
}

/* ────────────────────────── Train ────────────────────────── */
export const trainAndTuneTool: FunctionDeclaration = {
    name: 'train_and_tune',
    description:
        'Train and evaluate machine-learning models on the cleaned dataset. ' +
        'Specify the target column, which algorithms to try, the evaluation metric, and whether to maximise or minimise it.',
    parameters: {
        type: 'object' as any,
        properties: {
            target_column: {
                type: 'string' as any,
                description: 'Name of the column to predict.',
            },
            models: {
                type: 'array' as any,
                items: { type: 'string' as any },
                description:
                    'List of model identifiers to train. Supported: "knn", "decision_tree", "random_forest".',
            },
            metric: {
                type: 'string' as any,
                description:
                    'Evaluation metric. Supported: "accuracy", "f1", "precision", "recall", "mse", "r2".',
            },
            metric_direction: {
                type: 'string' as any,
                description: '"maximize" or "minimize".',
            },
        },
        required: ['target_column', 'models', 'metric', 'metric_direction'],
    },
}

/* ──────────────────── Aggregated list ──────────────────── */
export const allTools = [inspectDataTool, cleanDataTool, trainAndTuneTool]
