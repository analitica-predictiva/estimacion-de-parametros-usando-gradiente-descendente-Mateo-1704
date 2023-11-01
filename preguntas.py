"""
Optimización usando gradiente descendente - Regresión polinomial
-----------------------------------------------------------------------------------------

En este laboratio se estimarán los parámetros óptimos de un modelo de regresión 
polinomial de grado `n`.

"""


def pregunta_01():
    """
    Complete el código presentado a continuación.
    """
    # Importe pandas
    import pandas as pd

    # Importe PolynomialFeatures
    from sklearn.preprocessing import PolynomialFeatures

    # Cargue el dataset `data.csv`
    data = pd.read_csv("data.csv")

    # Cree un objeto de tipo `PolynomialFeatures` con grado `2`
    poly = PolynomialFeatures(2)

    # Transforme la columna `x` del dataset `data` usando el objeto `poly`
    x_poly = poly.fit_transform(data[["x"]])

    # Retorne x y y
    return x_poly, data.y


def pregunta_02():

    # Importe numpy
    import numpy as np

    x_poly, y = pregunta_01()

    # Fije la tasa de aprendizaje en 0.0001 y el número de iteraciones en 1000
    learning_rate = 0.0001
    n_iterations = 500

    # Defina el parámetro inicial `params` como un arreglo de tamaño 3 con ceros
    # params = np.zeros(np.shape[2])
    params = np.zeros(3)
    params = np.array([0.0,0.0,0.0])

    # print(params)
    x = [ elem[1] for elem in x_poly]
    #print( x)
    y_pred = np.polyval(params, x) 
    #print("y_pred", y_pred)

    for ind in range(n_iterations):

        # Compute el pronóstico con los parámetros actuales
        y_pred = np.polyval(params, x)

        # Calcule el error
        errors = y - y_pred
        error =  sum(e**2 for e in errors)


        # Calcule el gradiente
        gradient_w0 = -2 * sum(errors)
        gradient_w1 = -2 * sum(
                [e * x_value for e, x_value in zip(errors, x)]
            )
        gradient_w2 = -2 * sum(
                [e * x_value * x_value for e, x_value in zip(errors, x)]
            )
        
        gradient = np.array([gradient_w2,gradient_w1,gradient_w0])

        # Actualice los parámetros
        params = params - learning_rate * gradient

    return np.array([params[2,], params[1],params[0]])
