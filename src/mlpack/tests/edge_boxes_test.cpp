/**
 * @file edge_boxes_test.cpp
 * @author Nilay Jain
 *
 * Tests for functions in edge_boxes algorithm.
 */

#include <mlpack/core.hpp>
#include <map>
#include <mlpack/methods/edge_boxes/feature_extraction.hpp>

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

using namespace mlpack;
using namespace mlpack::structured_tree;
/*
  
 
  //void GetShrunkChannels(CubeType& InImage, CubeType& reg_ch, CubeType& ss_ch);
  
  CubeType RGB2LUV(CubeType& InImage);

  void Gradient(CubeType& InImage, 
                MatType& Magnitude,
                MatType& Orientation);

  
  CubeType Histogram(MatType& Magnitude,
                       MatType& Orientation, 
                       int downscale, int interp);
 
  /*
  CubeType ViewAsWindows(CubeType& channels, arma::umat& loc);

  CubeType GetRegFtr(CubeType& channels, arma::umat& loc);

  CubeType GetSSFtr(CubeType& channels, arma::umat& loc);

  CubeType Rearrange(CubeType& channels);

  CubeType PDist(CubeType& features, arma::uvec& grid_pos);
  ***-/
*/

BOOST_AUTO_TEST_SUITE(EdgeBoxesTest);

/**
 * This test checks the feature extraction functions 
 * mentioned in feature_extraction.hpp
 */

void Test(arma::mat m1, arma::mat m2)
{
  double* d1 = m1.memptr();
  double* d2 = m2.memptr();
  std::cout << arma::size(m1) << std::endl;
  std::cout << arma::size(m2) << std::endl;
  for (size_t i = 0; i < m1.n_elem; ++i)
    BOOST_REQUIRE_CLOSE(*d1, *d2, 1e-3);  
}

void Test(arma::cube m1, arma::cube m2)
{
  double* d1 = m1.memptr();
  double* d2 = m2.memptr();
  std::cout << arma::size(m1) << std::endl;
  std::cout << arma::size(m2) << std::endl;
  for (size_t i = 0; i < m1.n_elem; ++i)
    BOOST_REQUIRE_CLOSE(*d1, *d2, 1e-3);  
}

void DistanceTransformTest(arma::mat& input,
                          double on, arma::mat& output,
                          StructuredForests<arma::mat, arma::cube>& SF)
{
  arma::mat dt_output = SF.dt_image(input, on);
  Test(dt_output, output);
  std::cout << "DistanceTransformTest completed" << std::endl;
} 

arma::cube CopyMakeBorderTest(arma::cube& input,
                              arma::cube& output,
                          StructuredForests<arma::mat, arma::cube>& SF)
{
  arma::cube border_output = SF.CopyMakeBorder(input, 1, 1, 1, 1);
  Test(border_output, output);
  std::cout << "CopyMakeBorderTest completed" << std::endl;
}

arma::cube RGB2LUVTest(arma::cube& input, arma::cube& output,
                  StructuredForests<arma::mat, arma::cube>& SF)
{
  arma::cube luv = SF.RGB2LUV(input);
  Test(luv, output);
  std::cout << "RGB2LUVTest completed" << std::endl;
}

arma::cube ConvTriangleTest(arma::cube& input, int radius,
    arma::cube& output, StructuredForests<arma::mat, arma::cube>& SF)
{
  arma::cube conv_out = SF.ConvTriangle(input, radius);
  Test(conv_out, output);
  std::cout << "ConvTriangleTest passed" << std::endl;
}

BOOST_AUTO_TEST_CASE(FeatureExtractionTest)
{
  std::map<std::string, int> options;
  options["num_images"] = 2;
  options["row_size"] = 321;
  options["col_size"] = 481;
  options["rgbd"] = 0;
  options["shrink"] = 2;
  options["n_orient"] = 4;
  options["grd_smooth_rad"] = 0;
  options["grd_norm_rad"] = 4;
  options["reg_smooth_rad"] = 2;
  options["ss_smooth_rad"] = 8;
  options["p_size"] = 32;
  options["g_size"] = 16;
  options["n_cell"] = 5;

  options["n_pos"] = 10000;
  options["n_neg"] = 10000;
  options["n_tree"] = 8;
  options["n_class"] = 2;
  options["min_count"] = 1;
  options["min_child"] = 8;
  options["max_depth"] = 64;
  options["split"] = 0;  // we use 0 for gini, 1 for entropy, 2 for other
  options["stride"] = 2;
  options["sharpen"] = 2;
  options["n_tree_eval"] = 4;
  options["nms"] = 1;    // 1 for true, 0 for false

  arma::mat input, output;
  input << 0 << 0 << 0 << arma::endr
        << 0 << 1 << 0 << arma::endr
        << 1 << 0 << 0;
  
  output << 2 << 1 << 2 << arma::endr
         << 1 << 0 << 1 << arma::endr
         << 0 << 1 << 2;
  StructuredForests<arma::mat, arma::cube> SF(options);
  DistanceTransformTest(input, 1, output, SF);
  
  arma::cube in1(input.n_rows, input.n_cols, 1);
  arma::cube c1(input.n_rows, input.n_cols, 1);
  
  in1.slice(0) = output;
  
  arma::mat out_border;
  out_border << 2 << 2 << 1 << 2 << 2 << arma::endr
             << 2 << 2 << 1 << 2 << 2 << arma::endr
             << 1 << 1 << 0 << 1 << 1 << arma::endr
             << 0 << 0 << 1 << 2 << 2 << arma::endr
             << 0 << 0 << 1 << 2 << 2;
  arma::cube out_b(out_border.n_rows, out_border.n_cols, 1);
  out_b.slice(0) = out_border;
  CopyMakeBorderTest(in1, out_b, SF);

  std::cout << "reached here" << std::endl;
  arma::mat out_conv;

  out_conv << 1.20987 << 1.25925 << 1.30864 << arma::endr
           << 0.96296 << 1.11111 << 1.25925 << arma::endr
           << 0.71604 << 0.96296 << 1.20987;


  c1.slice(0) = out_conv;

  ConvTriangleTest(in1, 2, c1, SF);

  arma::mat out_luv;
  out_luv << 0.15777 << 0.39216 << 0.41656 << arma::endr
          << 0.25243 << 0.14790 << 0.71061 << arma::endr
          << 0.19001 << 0.56765 << 0.46887;

  arma::cube out_c(output.n_rows, output.n_cols, 3);
  arma::cube in_luv(output.n_rows, output.n_cols, 3);
  for(size_t i = 0; i < out_c.n_slices; ++i)
  {
    in_luv.slice(i) = output / 10;
    out_c.slice(i) = out_luv;
  }
  RGB2LUVTest(in_luv, out_c, SF);
  std::cout << "finish" << std::endl;
}

BOOST_AUTO_TEST_SUITE_END();
