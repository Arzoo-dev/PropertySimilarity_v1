# Auto Testing Results

## Configuration

- Model directory: siamese_resnet50_20250423_early/final_model
- Model type: siamese_embedding_model
- Test data: test_properties/
- Subject property: 4544a10a9d2d49f4a7d253e67b7abd37
- Number of comp properties: 5
- Similarity threshold: 5.0

## Summary

- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1 Score: 1.0000
- Average similarity score for similar pairs: 9.98
- Average similarity score for dissimilar pairs: 0.00

## Explanation of Metrics

- **TP (True Positive)**: 5 - Correctly identified as similar
- **FP (False Positive)**: 0 - Incorrectly identified as similar
- **TN (True Negative)**: 0 - Correctly identified as dissimilar
- **FN (False Negative)**: 0 - Incorrectly identified as dissimilar
- **Accuracy**: (TP+TN)/(TP+TN+FP+FN) - Proportion of correct predictions
- **Precision**: TP/(TP+FP) - Of all predicted similar, how many were actually similar
- **Recall**: TP/(TP+FN) - Of all actually similar, how many were correctly predicted
- **F1 Score**: 2*(precision*recall)/(precision+recall) - Harmonic mean of precision and recall

See `4544a10a9d2d49f4a7d253e67b7abd37_test_results.json` for detailed results for each comp property comparison.
