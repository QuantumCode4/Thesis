# Code_Aceptance

### Funciones auxiliares

La función `sample_theta_cosn` genera ángulos θ distribuidos de acuerdo con una distribución proporcional a $\cos^n(\theta)$ en el intervalo $[0, \pi/2]$, usando el método de rechazo.

La función `generate_muons` simula el paso de muones a través de un sistema de múltiples planos detectores, evaluando cuáles atraviesan todos los planos sin salirse del área activa del detector. Comienza generando $N$ muones con ángulos azimutales ($\phi$) uniformemente distribuidos y ángulos polares ($\theta$) muestreados según una distribución $\cos^n(\theta)$, lo que modela la distribución angular realista de muones cósmicos. A cada muón se le asigna una posición inicial aleatoria dentro del primer plano, de tamaño $L \times L$, y su trayectoria es proyectada hacia planos sucesivos separados por una distancia $D$, calculando su posición de impacto en cada uno según trigonometría. Cabe aclarar que el vector unitario de cada dirección está parametrizado de forma convencional con los ángulos polares. Si en algún plano el muón se sale de los límites del detector, se descarta. La función retorna los ángulos $\theta$ generados y una máscara booleana que indica qué muones fueron aceptados, es decir, aquellos que pasaron por todos los planos dentro del área activa.

### Parámetros

`N_MUONS` es el número de muones a simular, `L` es el largo del plano de detección, `D` es la separación de cada plano de detección, `N_planes` es el número de planos de deteccion, `NUM_BINS` el número de bins para la distribución de angulos $\theta$  y `n` es el exponente para la distrución $\cos^n(\theta)$. Con `L`, `D` y `N_planes` se calcula el ángulo $\theta$ máximo al cual los conteos deberían empezar a ser nulos sabiendo que, en el caso límite, el muon entra en una esquina del plano superior y sale en la esquina opuesta del plano inferior.

### Calculo de la Aceptancia

Primero, se construyen histogramas para contar cuántos muones fueron aceptados (`accepted_theta`) y cuántos fueron generados en total (`theta`) dentro de un conjunto de bins angulares equiespaciados entre 0 y $\pi/2$. Luego, se calcula la probabilidad de aceptación por bin dividiendo el número de aceptados entre el número de generados en cada bin. Esta división se realiza con `np.divide`, que incluye el argumento `where=counts_generated != 0` para evitar divisiones por cero y asignar cero en aquellos bins donde no se generaron muones. El resultado, `prob_acceptance_per_bin`, representa la aceptancia relativa del sistema.

### Plot

Se genera una gráfica de barras que representa la aceptancia angular del hodoscopio en función del ángulo $\theta$, expresado en grados. Primero, se calculan los centros de los bins en grados a partir de los bordes del histograma. Luego, se construye una figura con `matplotlib`, donde cada barra indica la fracción de muones aceptados para un intervalo angular específico.

### Descomposición de $\theta$

En el siguiente código se descompone el ángulo cenital $\theta$ en sus componentes x e y ($\theta_x$ y $\theta_y$), lo cual permite representar la aceptancia del detector en dos dimensiones. Usando las relaciones trigonométricas, se calcula $\theta_x$ como $\arctan(\tan(\theta)\cos(\phi))$ y $\theta_y$ como $\arctan(\tan(\theta)\sin(\phi))$, convirtiéndolos luego a grados. Esto equivale a proyectar el ángulo tridimensional $\theta$ en el plano x-z ($\theta_x$) y el plano y-z ($\theta_y$), utilizando el ángulo azimutal $\phi$ para separar sus componentes. Luego, se crean histogramas bidimensionales de los ángulos generados y aceptados para obtener la distribución de aceptancia en función de $\theta_x$ y $\theta_y$. Finalmente, se calcula la aceptancia 2D como la fracción de muones aceptados sobre los generados en cada celda angular.

Luego, se genera un mapa de calor 2D. Primero, se define el rango de los ejes (`extent`) usando los bordes de los bins en grados. Luego, `imshow` se utiliza para visualizar la matriz `acceptance_2D.T`, que contiene la fracción de muones aceptados por bin en el espacio angular proyectado. La transposición `.T` es necesaria para que los ejes del gráfico coincidan con θₓ en el eje horizontal y θᵧ en el vertical.
