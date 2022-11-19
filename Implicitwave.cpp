/*A simple code for u_tt = c^2 u_xx with u[t=0,x] = sin(4pix/L), u_t[0,x]= 0 and u[t,x=0] = 0 =u[t, x=L]  x: [0,L=1], t: [0, T]  using first Explicit and than Implicit scheme*/

#include <iostream>
#include <math.h> //use for M_PI
using namespace std;


/*following book by Titus Beu */
double *Vector(int imin, int imax)
{
   double *p; // assign block start to array pointer
   p = (double*) malloc((size_t) ((imax-imin+1)*sizeof(double)));
   if (!p) {
      printf("Vector: allocation error !\n");
      exit(1);
   }
   return p - imin; //adjust for offset
}


void TDMA(double a[], double b[], double c[], double d[], int n) 
//Solves a system with tridiagonal matrix by LU factorization (diag(L) = 1).
//a - lower codiagonal (i=2,n)
//b - main diagonal (i=1,n)
//c - upper codiagonal (i=1,n-1)
// d - constant terms (i=1,n); solution on exit
//n - order of system.
{
    int i;
    if (b[1] == 0e0) 
        { printf("TriDiagSys: singular matrix !\n"); return; }
    for (i=2; i<=n; i++) 
       {                                    // factorization
        a[i] /= b[i-1];
        b[i] -= a[i]*c[i-1];
        if (b[i] == 0e0) 
	    { printf("TriDiagSys: singular matrix !\n"); return; }
        d[i] -= a[i]*d[i-1];
       }
    d[n] /= b[n];                                     // backward substitution
    for (i=(n-1); i>=1; i--) 
        d[i] = (d[i] - c[i]*d[i+1])/b[i];
}


void ImplicitInit(double u0[], double u1[], double a[], double b[], double c[], double d[], double x[], int nx, double c, double dx, double dt)
//---------------------------------------------------------------------------
//Not working at the moment
{
   double coeff, coeff2, sqcoeff, v0;
   coeff = c*dt/dx; sqcoeff = coeff*coeff;               
   coeff2 = 1e0 + sqcoeff;
   int i;
   for (i=1; i<=nx; i++){          // time step 0
      v0 = 0e0;
      u0[i] = sin(4*M_PI*x[i]/1.0);
      a[i] = -0.5*sqcoeff;
      c[i] = -0.5*sqcoeff;
      d[i] = u0[i] + dt * v0;
       }
   for (i=1; i<=nx+1; i++){
       b[i] = coeff2;
       }
   TDMA(a, b, c, d, nx)
   u1[1] = u0[1]; u1[nx] = u0[nx];
   for (i=1; i<=nx; i++){
         u1[i] = d[i];}
}

void Implicitpropagte(double um1[], double u0[], double u1[], double a[], double b[], double c[], double d[], double x[], int nx, double c, double dx, double dt)
//---------------------------------------------------------------------------
//Not working at the moment
{
   double coeff, coeff2, sqcoeff;
   coeff = c*dt/dx; sqcoeff = coeff*coeff;                  // time step 1
   coeff2 = 1e0 + 2* sqcoeff;
   int i;
   for (i=1; i<=nx; i++){

      a[i] = -sqcoeff;
      c[i] = -sqcoeff;
      d[i] = 2u0[i] - + um1[i];
       }
   for (i=1; i<=nx+1; i++){
       b[i] = coeff2;
       }
   TDMA(a, b, c, d, nx)
   u1[1] = u0[1]; u1[nx] = u0[nx];
   for (i=1; i<=nx; i++){
         u1[i] = d[i];}

}

int main()
{
   double *u0, *u1, *u, *x, *a, *b,*c, *d;
   double Cc, dt, dx, t, tmax, xmax, cflcoeff, v0;
   int i, it,  nout, nt, nx;
   char fname[80], title[80];
   FILE *out;
   Cc    = 1e0;         /*speed of wave*/
   xmax = 1e0;          /*maximum x =L=1.0*/
   tmax = 40e0;         /*maximum propagation time*/
   dt   = 2.5e-3;       /*time step*/
   cflcoeff = 0.25;     /*CFL which  we use > 1 to test implicit scheme*/
   dx  = Cc*dt/cflcoeff;    // spatial step size
   nout = 500;

      nx = 1*(int)(xmax/dx + 0.5) + 1;   // odd number of spatial nodes
   nt = (int)(tmax/dt + 0.5);                         // number of time steps
   nx2 = nx/2;

   u0 = Vector(1,nx);                                     // initial solution
   u1 = Vector(1,nx);    // first propagated solution
   u  = Vector(1,nx);                                             // solution
   x  = Vector(1,nx);                                         // spatial mesh
   a  = Vector(1,nx);                                         // spatial mesh
   b  = Vector(1,nx);                                         // spatial mesh
   c  = Vector(1,nx);                                         // spatial mesh
   d  = Vector(1,nx);     // spatial mesh
   /*cout << "dx = " << dx;*/
   for (i=1; i<=nx; i++)
      {x[i] = (i-1)*dx ;
       cout << "xi  =  " << x[i];}
  /*Init(u0, u1, a, b, c, d, x, nx, Cc, dx, dt);*/
   /*stepI is to get u0*/
  for (i=1; i<=nx; i++){        // time step 0
      v0 = 0e0;
           u0[i] = sin(4*M_PI*x[i]/1.0);}
  /*get u1 solvingtridiagnoal system*/
    a[1] = 0e0;
   for (i=2; i<=nx; i++){        // time step 0
      a[i] = -0.5*cflcoeff*cflcoeff;}

   for (i=1; i<=nx-1; i++){
      c[i] = -0.5*cflcoeff*cflcoeff;}
      c[nx] = 0e0;
   for (i=1; i<=nx; i++){        // time step 0
      d[i] = sin(4*M_PI*x[i]/1.0) + dt * v0;
      b[i] = 1.0 + cflcoeff*cflcoeff;
   }
   TDMA(a, b, c, d, nx);
   /*u1[1] = u0[1]; u1[nx] = u0[nx];*/
   for (i=1; i<=nx; i++){
         u1[i] = d[i];}
   for (it=1; it<=nt; it++){
     t = it * dt;
    //printf("TriDiagSys: singular matrix %d !\n", it);
    Implicitpropagte(u0, u1, u, a, b, c, d, x, nx,Cc, dx, dt);    // shift solutions

    for (i=1; i<=nx; i++){
             u0[i] = u1[i];
             u1[i] = u[i];
             u[i] = d[i];}
    if (it % nout == 0 || it == nt) {
         printf("TriDiagSys: !\n");
         sprintf(fname,"outimplicitwave_%4.2f.txt",t);
         out = fopen(fname,"w");
         fprintf(out,"t = %4.2f\n",t);
         fprintf(out,"     x            u\n");
         for (i=1; i<=nx; i++)
            fprintf(out,"%10.5f%10.5f\n",x[i],u[i]);
         fclose(out);
      }
   }
}

