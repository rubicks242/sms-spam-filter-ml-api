## SMS Spam Filter

### Endpoint:

> /predict [POST]

### Request Body:

> { "message" : "Wow! you won $100k" }

### Response Body:

> {
    "decision_tree": "ham",
    "logistic_regression": "spam",
    "naive_bayes": "spam",
    "svc": "spam"
}
