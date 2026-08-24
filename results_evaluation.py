from sklearn.metrics import classification_report


def compute_metrics(y_true, y_pred, set):
    def get_flat_dict(d, parent_key='', sep= '_'):
        items = {}
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(get_flat_dict(v, new_key, sep=sep))
            else:
                items[new_key] = v
        return items

    metrics = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    # filtering single class metrics
    filtered_metrics = {k: v for k, v in metrics.items() if not k[0].isdigit()}
    flatten_metrics = get_flat_dict(filtered_metrics)
    flatten_metrics ={f'{set}_{k}': round(v*100, 2) for k, v in flatten_metrics.items() if 'support' not in k}

    return flatten_metrics
