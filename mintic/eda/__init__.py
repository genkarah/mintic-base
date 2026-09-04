import numpy as np



## FUNCION 1

def impute_missing(data, strategy="mean", columns=None):

    # Veo metieron bien strategy
    if strategy not in ("mean", "median", "mode"):
        raise ValueError(
            "strategy debe ser 'mean', 'median', or 'mode'"
        )

    
    result = data.copy()

    # Veo cuales cols si se procesaran
    if columns is None:
        # La media y mediana solo se aplican a datos num
        if strategy in ("mean", "median"):
           selected_columns = [
            column
            for column in result.columns
            if result[column].dtype.kind in "biufc"  #b = booleanos, i= enteros, u = enteros sin signp, f = num decimal y c= num complejo
        ]

        #moda
        else:
            selected_columns = list(result.columns)

    # pasamos una columna como texto 
    elif isinstance(columns, str):
        selected_columns = [columns]

    # Si columns ya es una lista, hacemos una copia de ella.
    else:
        selected_columns = list(columns)

    # la columna existe?
    missing_columns = [
        column
        for column in selected_columns
        if column not in result.columns
    ]

    if missing_columns:
        raise KeyError(f"No encontre esa columna: {missing_columns}")

    # Procesamos una columna a la vez.
    for column in selected_columns:

        # Si la columna no tiene valores faltantes, continuamos.
        if not result[column].isna().any():
            continue

        # Eliminamos temporalmente los faltantes para hacer las operaciones
        values = result[column].dropna().to_numpy()

        # No podemos calcular una estrategia si toda la columna está vacía.
        if values.size == 0:
            raise ValueError(
                f"La columna '{column}' solo contiene valores faltantes :("
            )

        # Calculamos el valor que reemplazara a los faltantes
        if strategy == "mean":
            if result[column].dtype.kind not in "biufc":
                raise TypeError(
                    f"La funcion mean requiere valores numericos: '{column}'"
                )

            replacement = np.mean(values)

        elif strategy == "median":
            if result[column].dtype.kind not in "biufc":
                raise TypeError(
                    f"La funcion median requiere valores numericos: '{column}'"
                )

            replacement = np.median(values)

        else:
            unique_values, counts = np.unique(
                values,
                return_counts=True
            )

            replacement = unique_values[np.argmax(counts)]

        # Sustituimos 
        result[column] = result[column].fillna(replacement)

    
    return result





## FUNCION 2

def detect_outliers(data, method="iqr", threshold=1.5):
    

    # Validar 
    if method not in ("iqr", "zscore"):
        raise ValueError(
            "metodo debe ser 'iqr' o 'zscore'"
        )

    # El umbral debe ser positivo
    if threshold <= 0:
        raise ValueError(
            "el umbral tiene que ser positivo"
        )

    # Crear un df booleano del mismo tamaño que data
    outliers = data.isna()
    outliers.iloc[:, :] = False

    
    for column in data.columns:

        # Ignorar las columnas que no sean num
        if data[column].dtype.kind not in "iuf":
            continue

        # Obtener los valores no faltantes como arreglo 
        values = data[column].dropna().to_numpy()

        # Si no existen valores disponibles, pasar a otra columna
        if values.size == 0:
            continue

        if method == "iqr":
            # Calcular el primer y tercer cuartil.
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)

            
            iqr = q3 - q1

            # Calcular los limites
            lower_limit = q1 - threshold * iqr
            upper_limit = q3 + threshold * iqr

            # Marcar lo vals que esten afuera de eso
            outliers[column] = (
                (data[column] < lower_limit)
                | (data[column] > upper_limit)
            )

        else:
            
            mean = np.mean(values)
            standard_deviation = np.std(values)
            
            if standard_deviation == 0:
                continue

            z_scores = np.abs(
                (data[column] - mean)
                / standard_deviation
            )

            # Marcar valores cuyo z-score supere el umbral
            outliers[column] = z_scores > threshold

    return outliers