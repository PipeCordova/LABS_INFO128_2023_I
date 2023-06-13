/* Integrantes:
	- Felipe Córdova
	- Sebastián Montecinos
	- Diego Vidal
*/

#include <iostream>
#include <omp.h>
using namespace std;

// Definir funciones
void imprimirArreglo(int arreglo[], int tam, string msj);
int* gap_sum(int arreglo[], int tam, int gap);

int main(int argc, char* argv[]){
    if (argc != 3) {
        cout << "Se tiene que ejecutar ./prog n gap." << endl;
        return EXIT_FAILURE;
    }
    // Semilla fija
    unsigned int semilla = 12345;
    srand(semilla);

    // Definir variables
    int n =  atoi(argv[1]);
    int gap = atoi(argv[2]);
    int tam = n+gap*(n-1);
    int *array = new int[tam];
    bool flg = false;

    // Crea el arreglo inicial
    int indice = 0;
    while (indice < tam) {
        array[indice] = 1 + rand()%(1000);
        indice += gap + 1;
    }
    cout << "Arreglo inicial creado con exito..." << endl;

    if(flg)
        imprimirArreglo(array, tam, "inicial");
	
    // Crea el arreglo final y mide el tiempo de ejecucion
    double t1 = omp_get_wtime();
    array = gap_sum(array, tam, gap);
    double t2 = omp_get_wtime();
	double tiempo = t2 - t1;

    if(flg)
        imprimirArreglo(array, tam, "final  ");

    // Imprimir tiempo de ejecucion
    cout << "El tiempo de ejecucion fue: " << tiempo << " segundos." << endl;

    // Liberar espacio en memoria
    delete[] array;
    cout << "FIN LAB-05!!\n";
    return EXIT_SUCCESS;
}
/* Funcion creada para imprimir. Se recomienda usar cuando la entrada sea pequeña para comprobar el trabajo.
Si la entrada es grande no se recomienda utilizar. Se entregará el codigo con un flg inicialmente en false, si desea usar 
esta funcion cambie el boleano del main a true (linea 29) y verá en pantalla el resultado del arreglo. */
void imprimirArreglo(int arreglo[], int tam, string msj){
    cout << "Arreglo " << msj << " : ";
    for (int i = 0; i < tam; i++)
        cout << arreglo[i] << " ";
    cout << endl;
	return;
}

//Esta funcion retorna el arreglo final
int* gap_sum(int arreglo[], int tam, int gap) {
    int sum = 0;
    for(int pos=(tam-1) ; pos >= 0 ; pos = pos - (gap+1)){
        sum += arreglo[pos];
        arreglo[pos] = sum;
    }
    return arreglo;
}
