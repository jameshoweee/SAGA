#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <stdlib.h>

/*
 * Sample from the half-Gaussian distribution using the RCDT (reverse CDT).
 *
 * The table below is the RCDT of the half-Gaussian with sigma_0 = 1.8205,
 * matching Falcon's reference implementation. Each row is a 72-bit threshold
 * stored as three 24-bit limbs (big-endian within the row).
 *
 * The sampler draws a uniform 72-bit value and counts how many thresholds
 * it exceeds; the count is the output z0 in {0, ..., 18}.
 */

int gaussian0()
{
    static const uint32_t rcdt[] = {
         6031371U, 13708371U, 13035518U,
        11218132U, 15196352U,  8529022U,
        14516786U,  3108023U, 14040577U,
        16068234U, 12355640U,  6731036U,
        16607867U,  9654540U, 12640401U,
        16746677U,  3713810U,  9126561U,
        16773083U,  2272212U,  8951068U,
        16776798U,     9114U,  5413926U,
        16777184U,  8333173U,  8690648U,
        16777214U,  3932749U, 16511895U,
        16777215U, 15544539U,  3132933U,
        16777215U, 16739168U,  7665377U,
        16777215U, 16776345U, 10638952U,
        16777215U, 16777201U,  4231493U,
        16777215U, 16777215U, 13673090U,
        16777215U, 16777215U, 16748392U,
        16777215U, 16777215U, 16777018U,
        16777215U, 16777215U, 16777215U
    };

    uint32_t v0, v1, v2;
    size_t u;
    int z;

    v0 = lrand48() & 0xFFFFFF;
    v1 = lrand48() & 0xFFFFFF;
    v2 = lrand48() & 0xFFFFFF;

    z = 0;
    for (u = 0; u < (sizeof rcdt) / sizeof(rcdt[0]); u += 3) {
        uint32_t w0, w1, w2, cc;

        w0 = rcdt[u + 2];
        w1 = rcdt[u + 1];
        w2 = rcdt[u + 0];
        cc = (v0 - w0) >> 31;
        cc = (v1 - w1 - cc) >> 31;
        cc = (v2 - w2 - cc) >> 31;
        z += (int)cc;
    }
    return z;
}


int main()
{
    FILE *f = fopen("samples.txt", "a");
    int b, z0, z;
    int sample_size = 1000000;

    srand48(time(NULL));

    for (int i = 0; i < sample_size; i++) {
        z0 = gaussian0();
        b = lrand48() & 1;
        z = b + ((b << 1) - 1) * z0;

        printf("%d\n", z);
        fprintf(f, "%d\n", z);
    }
    fclose(f);

    return 0;
}
