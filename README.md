## SMS Spam Filter

### Endpoint:

> /predict [POST]

> /message-count [GET]

### Request Body:

> { "message" : "Wow! you won $100k" }

### Response Body:

> {
    "decision_tree": "ham",
    "logistic_regression": "spam",
    "naive_bayes": "spam",
    "svc": "spam"
}

>   {
        'spam_count': 747,
        'ham_count': 4573
    }

### Example usage:

GITHUB_URL: https://github.com/mssandeepkamath/sms-spam-filter-android.git

