# -*- coding: utf-8 -*-
"""
Python 3
05 / 07 / 2024
@author: z_tjona

"I find that I don't understand things unless I try to program them."
-Donald E. Knuth
"""

# ----------------------------- logging --------------------------
import logging
from sys import stdout
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    stream=stdout,
    datefmt="%m-%d %H:%M:%S",
)
logging.info(datetime.now())

import numpy as np


# ####################################################################
def eliminacion_gaussiana(A: np.ndarray) -> np.ndarray:
    """Resuelve un sistema de ecuaciones lineales mediante el método de eliminación gaussiana.

    ## Parameters

    ``A``: matriz aumentada del sistema de ecuaciones lineales. Debe ser de tamaño n-by-(n+1), donde n es el número de incógnitas.

    ## Return

    ``solucion``: vector con la solución del sistema de ecuaciones lineales.

    """
    if not isinstance(A, np.ndarray):
        logging.debug("Convirtiendo A a numpy array.")
        A = np.array(A)
    assert A.shape[0] == A.shape[1] - 1, "La matriz A debe ser de tamaño n-by-(n+1)."
    n = A.shape[0]

    for i in range(0, n - 1):  # loop por columna

        # --- encontrar pivote
        p = None  # default, first element
        for pi in range(i, n):
            if A[pi, i] == 0:
                # must be nonzero
                continue

            if p is None:
                # first nonzero element
                p = pi
                continue

            if abs(A[pi, i]) < abs(A[p, i]):
                p = pi

        if p is None:
            # no pivot found.
            raise ValueError("No existe solución única.")

        if p != i:
            # swap rows
            logging.debug(f"Intercambiando filas {i} y {p}")
            _aux = A[i, :].copy()
            A[i, :] = A[p, :].copy()
            A[p, :] = _aux

        # --- Eliminación: loop por fila
        for j in range(i + 1, n):
            m = A[j, i] / A[i, i]
            A[j, i:] = A[j, i:] - m * A[i, i:]

        logging.info(f"\n{A}")

    if A[n - 1, n - 1] == 0:
        raise ValueError("No existe solución única.")

        print(f"\n{A}")
    # --- Sustitución hacia atrás
    solucion = np.zeros(n)
    solucion[n - 1] = A[n - 1, n] / A[n - 1, n - 1]

    for i in range(n - 2, -1, -1):
        suma = 0
        for j in range(i + 1, n):
            suma += A[i, j] * solucion[j]
        solucion[i] = (A[i, n] - suma) / A[i, i]

    return solucion


# ####################################################################
def descomposicion_LU(A: list[list[float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Realiza la descomposición LU de una matriz A con pivoteo parcial.

    Parameters
    ----------
    A : list[list[float]]
        Matriz cuadrada a descomponer.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        - P: Matriz de permutación.
        - L: Matriz triangular inferior con diagonal principal de 1's.
        - U: Matriz triangular superior.

    Raises
    ------
    ValueError
        Si la matriz no es cuadrada.
    """
    # Convertir a matriz NumPy
    A = np.array(A, dtype=float)

    # Dimensiones de la matriz
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("La matriz debe ser cuadrada.")

    # Inicialización de P, L y U
    P = np.eye(n)  # Matriz de permutación inicial
    L = np.zeros_like(A)
    U = A.copy()

    for i in range(n):
        # Pivoteo parcial: buscar el elemento máximo en la columna i
        max_row = np.argmax(np.abs(U[i:, i])) + i
        if i != max_row:
            # Intercambiar filas en U
            U[[i, max_row], :] = U[[max_row, i], :]
            # Intercambiar filas en P
            P[[i, max_row], :] = P[[max_row, i], :]
            # Intercambiar filas en L, pero solo las columnas hasta i
            if i > 0:
                L[[i, max_row], :i] = L[[max_row, i], :i]

        # Actualizar L y U
        for j in range(i + 1, n):
            factor = U[j, i] / U[i, i]
            L[j, i] = factor
            U[j, i:] -= factor * U[i, i:]

    # L debe tener 1's en la diagonal principal
    np.fill_diagonal(L, 1)

    return P, L, U

# ####################################################################
def resolver_LU(L: np.ndarray, U: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Resuelve un sistema de ecuaciones lineales mediante la descomposición LU.

    ## Parameters

    ``L``: matriz triangular inferior.

    ``U``: matriz triangular superior.

    ``b``: vector de términos independientes.

    ## Return

    ``solucion``: vector con la solución del sistema de ecuaciones lineales.

    """

    n = L.shape[0]

    # --- Sustitución hacia adelante
    logging.info("Sustitución hacia adelante")

    y = np.zeros((n, 1), dtype=float)

    y[0] = b[0] / L[0, 0]

    for i in range(1, n):
        suma = 0
        for j in range(i):
            suma += L[i, j] * y[j]
        y[i] = (b[i] - suma) / L[i, i]

    logging.info(f"y = \n{y}")

    # --- Sustitución hacia atrás
    logging.info("Sustitución hacia atrás")
    sol = np.zeros((n, 1), dtype=float)

    sol[-1] = y[-1] / U[-1, -1]

    for i in range(n - 2, -1, -1):
        logging.info(f"i = {i}")
        suma = 0
        for j in range(i + 1, n):
            suma += U[i, j] * sol[j]
        logging.info(f"suma = {suma}")
        logging.info(f"U[i, i] = {U[i, i]}")
        logging.info(f"y[i] = {y[i]}")
        sol[i] = (y[i] - suma) / U[i, i]

    logging.debug(f"x = \n{sol}")
    return sol


# ####################################################################
def matriz_aumentada(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Construye la matriz aumentada de un sistema de ecuaciones lineales.

    ## Parameters

    ``A``: matriz de coeficientes.

    ``b``: vector de términos independientes.

    ## Return

    ``a``:

    """
    if not isinstance(A, np.ndarray):
        logging.debug("Convirtiendo A a numpy array.")
        A = np.array(A, dtype=float)
    if not isinstance(b, np.ndarray):
        b = np.array(b, dtype=float)
    assert A.shape[0] == b.shape[0], "Las dimensiones de A y b no coinciden."
    return np.hstack((A, b.reshape(-1, 1)))


# ####################################################################
def separar_m_aumentada(Ab: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Separa la matriz aumentada en la matriz de coeficientes y el vector de términos independientes.

    ## Parameters
    ``Ab``: matriz aumentada.

    ## Return
    ``A``: matriz de coeficientes.
    ``b``: vector de términos independientes.
    """
    logging.debug(f"Ab = \n{Ab}")
    if not isinstance(Ab, np.ndarray):
        logging.debug("Convirtiendo Ab a numpy array")
        Ab = np.array(Ab, dtype=float)
    return Ab[:, :-1], Ab[:, -1].reshape(-1, 1)
