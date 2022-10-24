import numpy as npy

"""
TRABAJO PRÁCTICO 3
 
Integrantes:
Perez Martin, Santiago (VINF010750)
Manzanares, Johana (VINF01938)
Llensa, Santiago (VINF09201)
Lobo, Carlos Ignacio (VINF010689)
Mariano, Martin Lorenzo (VINF09714) 
"""

def __PrintMatriz(lstIn: list):
    """
    Mostrar en pantalla
    Parameters
    ----------
    lstIn

    Returns
    -------

    """
    i = list
    for i in lstIn:
        temp = ''
        for j in i:
            if j == -1:
                temp += ' - '
            else:
                temp += ' X '
        print(temp)
    print('________________________________')


def __GenerarVector(matriz):
    """
    Genera vector a partir de la matriz
    Parameters
    ----------
    matriz

    Returns
    -------

    """
    tempVector = []
    temp = []
    for i in matriz:
        for j in i:
            temp.append(j)

    tempVector.append(temp)
    return tempVector


def CargarFaseAlmHebb(lstIn: list, intFase: int):
    '''
    Entreamiento fase con metodo Hebb
    Parameters
    ----------
    lstIn
    intFase

    Returns
    -------

    '''
    print(">>>>>> Patron nro. " + str(intFase))
    __PrintMatriz(lstIn)
    matrizImagen = npy.array(lstImagen, dtype=int)
    vectorPatron = npy.array(__GenerarVector(matrizImagen), dtype=int)
    matrizProduct = npy.dot(vectorPatron.T, vectorPatron)  # Producto

    return matrizProduct - npy.identity(100, dtype=int)


def CargarFaseAlmPseudoInv(lstIn: list, intFase: int):
    '''
    Entreamiento fase con metodo Hebb
    Parameters
    ----------
    lstIn
    intFase

    Returns
    -------

    '''
    print(">>> Carga FASE Nro. " + str(intFase))
    __PrintMatriz(lstIn)
    matrizImagen = npy.array(lstImagen, dtype=int)
    matrizPseudoInversa = npy.linalg.pinv(matrizImagen)  # Producto
    matrizProduct = npy.dot(matrizImagen, matrizPseudoInversa)

    return matrizProduct


def FaseRecuperacion(W: npy.array, vectorIn: npy.array, vectorOut: npy.array, intento: int):
    """
    Función recursiva para busqueda coincidencia
    Parameters
    ----------
    W = Matriz pesos
    vectorIn = Vector entrada
    vectorOut = Vector Salida
    intento = Numero de pasos

    Returns
    -------

    """
    def __FuncionActivacion(vector):
        vectorActivado = []
        listTemp = []
        for i in vector[0]:

            if i < 0:
                k = -1
            elif i == 0:
                k = 0
            else:
                k = 1
            vectorActivado.append(k)

        listTemp.append(vectorActivado)
        return npy.array(npy.array(listTemp))

    def __GenerarMatriz(vector, columnas: int):

        matrizTemp = []
        listTemp = []
        j = 0
        for i in vector:
            for k in i:
                j += 1
                listTemp.append(k)
                if j == columnas:
                    j = 0
                    matrizTemp.append(listTemp)
                    listTemp = []
        __PrintMatriz(matrizTemp)

    vectorOut = npy.dot(vectorIn, W)

    vectorOut = __FuncionActivacion(vectorOut)

    print(">>>>>> Fase Recuperacion Nro." + str(intento))
    __GenerarMatriz(vectorOut, 10)


    if (vectorIn == vectorOut).all():
        return vectorOut
    else:
        intento += 1
        vectorOut = FaseRecuperacion(W, vectorOut, 0, intento)


# ENTRENAMIENTO - Metodo Hebb

print(">>> FASE ENTRENAMIENTO - METODO HOPFIELD-HEBB")
# w1 = Matriz de pesos - 1er. Patron
lstImagen = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
[-1, -1, -1, -1, -1, -1, 1, -1, -1, -1],
[-1, -1, -1, -1, -1, 1, 1, 1, -1, -1],
[-1, -1, -1, -1, 1, 1, -1, 1, 1, -1],
[-1, -1, -1, 1, 1, -1, -1, -1, 1, 1],
[-1, -1, -1, -1, 1, 1, -1, 1, 1, -1],
[-1, -1, -1, -1, -1, 1, 1, 1, -1, -1],
[-1, 1, -1, -1, -1, -1, 1, -1, -1, -1],
[-1, 1, 1, -1, -1, -1, -1, -1, -1, -1],
[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
]
w1 = CargarFaseAlmHebb(lstImagen, 1)

# W = Almacenamiento(sumatoria de matrices de pesos de patrones)
W = w1

# VERIFICACION - Reconocimiento de patrones

lstPrueba = [[1, -1, -1, 1, -1, -1, -1, -1, -1, -1],
[-1, 1, -1, 1, -1, -1, 1, -1, 1, 1],
[1, -1, -1, -1, -1, 1, 1, -1, -1, 1],
[-1, 1, -1, -1, -1, 1, -1, -1, 1, -1],
[-1, -1, -1, 1, -1, -1, -1, -1, 1, -1],
[-1, -1, -1, -1, 1, 1, -1, 1, 1, -1],
[-1, -1, -1, -1, -1, -1, 1, -1, -1, -1],
[1, 1, -1, -1, -1, -1, 1, -1, -1, -1],
[-1, 1, 1, 1, 1, -1, -1, -1, 1, -1],
[-1, -1, -1, -1, -1, -1, -1, -1, -1, 1]
]
print(">>> FASE RECONOCIMIENTO ")
print(">>>>>> Imagen con ruido")
__PrintMatriz(lstPrueba)

matrizImagen = npy.array(lstPrueba, dtype=int)
vectorVerificacion = npy.array(__GenerarVector(matrizImagen), dtype=int)

FaseRecuperacion(W, vectorVerificacion, 0, 1)