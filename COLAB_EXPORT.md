# Export your actual trained models from Google Colab

Run this cell **in your existing Colab notebook** after Step 27. `model` is the first three-class Random Forest; `escalation_model` is the retrained binary model.

```python
import joblib
from google.colab import files

joblib.dump(model, 'risk_model.joblib')
joblib.dump(escalation_model, 'escalation_model.joblib')

files.download('risk_model.joblib')
files.download('escalation_model.joblib')
```

Place both downloaded files in the `models/` directory of the unzipped Streamlit project. **Do not retrain or replace them with fake placeholder models.**

## Final held-out evaluation (recommended before deployment)

```python
from sklearn.metrics import classification_report, confusion_matrix

final_scores = escalation_model.predict_proba(X_es_test)[:, 1]
final_pred = (final_scores >= 0.60).astype(int)
print(classification_report(y_es_test, final_pred, zero_division=0))
print(confusion_matrix(y_es_test, final_pred))
```

Note: This evaluation applies to the retrained escalation model and the validation-selected threshold. It should not be confused with results from the earlier model at threshold 0.50.
