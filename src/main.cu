#include <render.hpp>

int main(int argc, char const *argv[])
{   
    int shift = 17;
    float soft = 1e-7f;
    float step = 1e-7f;
    if(argc >= 5){
        step = powf(10, atof(argv[4]));
    }
    if(argc >= 4){
        soft = powf(10,atoi(argv[3])); 
    }

    if(argc >= 3){
        shift = atoi(argv[2]);
    }
    if(argc >= 2){
        if(argv[1][0] == 'g'){
            return renderG(1 << shift, soft,step);
        }
        else if(argv[1][0] == 'e'){
            return renderE(1 << shift, soft,step);
        }
    }
    
    return 0;
    
}

