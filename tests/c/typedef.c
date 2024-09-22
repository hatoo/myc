int printf(char *format, ...);

int main(void) {
    typedef int int_t;
    int_t v_int_t = 0;
    printf("%ld\n", sizeof(int_t));
    printf("%ld\n", sizeof(v_int_t));

    typedef int (*int_ptr_t)(void);
    int_ptr_t v_int_ptr_t = 0;
    printf("%ld\n", sizeof(int_ptr_t));
    printf("%ld\n", sizeof(v_int_ptr_t));

    struct s {
        int a;
        int b;
    };
    typedef struct s s_t;
    s_t v_s_t;
    printf("%ld\n", sizeof(s_t));
    printf("%ld\n", sizeof(v_s_t));

    union u {
        int a;
        int b;
    };
    typedef union u u_t;
    u_t v_u_t;
    printf("%ld\n", sizeof(u_t));
    printf("%ld\n", sizeof(v_u_t));

    return 0;
}