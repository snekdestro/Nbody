#include <render.hpp>

int main(int argc, char const *argv[])
{   
    if(argc >= 2){
        if(argv[1][0] == 'g'){
            return renderG();
        }
        else if(argv[1][0] == 'e'){
            return renderE();
        }
    }
    return 0;
    
}

