#ifndef TETGEN_UTIL_H
#define TETGEN_UTIL_H


#include <cmath>
#include <iostream>
#include <map>
#include <stack>
#include <tuple>
#include <vector>

#include "tetgen.h"


class TetgenVolumeMesh
{
private:
  
public:
  
  static void build_volume_mesh(std::vector<std::vector<int>> facets,std::vector<double> vertices){

    for (size_t i = 0; i < facets.size(); i++)
    {
      
      std::cout<< "Facet "<< i << "with " << facets[i].size() << " vertices." <<std::endl;
      std::cout<< " Vertices : ";
      for( const auto v :facets[i]){
        std::cout << "v: " << v << " ";
      }
      std::cout << std::endl;
    }
    
  }
};




#endif