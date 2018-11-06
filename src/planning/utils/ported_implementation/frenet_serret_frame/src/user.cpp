/* This file is not meant for production.
The purpose of the file is to demonstrate basic usage of the C++ API, and provide some sanity check.
*/

#include <algorithm>
#include <vector>
#include <limits>
#include <cmath>

#include <array>
// Include the exposed API for the module.
#include "FrenetSerretAPI.h"

#include "injectionData.h"

#include <iostream>
#include <chrono>


// Names aliasing:
using Single = float;
template<typename T>					  
using FrenetSerret2D		 = GM::UC::FrenetSerret::Frame<T, 2U, std::allocator<T>>;
template<std::uint32_t TnumberDimensions, std::uint32_t TnumberComponensInMainAxis> 
using PointsSingle			 = GM::UC::FrenetSerret::Points<float, TnumberDimensions, TnumberComponensInMainAxis>;
using Points2DSingleLocation = GM::UC::FrenetSerret::Points<float, 2U, 2U>; // Note that other configurations may not be supported (compile time indication.).
using Points2DSingleState    = GM::UC::FrenetSerret::Points<float, 2U, 6U>; // Note that other configurations may not be supported (compile time indication.).

using Single2D = std::array<Single, 2U>;


using micro = std::chrono::microseconds;





template<typename TdataType>
using ProjectedCartesianPoint = GM::UC::FrenetSerret::MATH_UTILITIES::ProjectedCartesianPoint<TdataType>;


/*
   Demonstrate transformations of points & states:
    C cartesian.
	F Frenet.
	Example: CFC   cartesian => frenent => cartesian.

*/
template<typename T>
bool casePointsCFC(FrenetSerret2D<T>* frame, int numberSamples);

template<typename T>
bool caseStateCFC(FrenetSerret2D<T>* frame, int numberSamples);


int main()
{
	// Inject points_in
	std::uint32_t pointsInDims[2U] = {5U, 2U};
	std::uint32_t statesInDims[2U] = {5U, 6U};
	std::uint32_t fStatesInDims[2U] = {5U, 6U};


	auto pointsInData = new Single[sizeof(pointsInInjectionData) / sizeof(pointsInInjectionData[0])];
	// transpose.
	for (auto i = int(); i < int(pointsInDims[0U]); ++i) 
	{
		pointsInData[i] = Single(pointsInInjectionData[2 * i]);
		pointsInData[i + int(pointsInDims[0U])] = Single(pointsInInjectionData[2 * i + 1]);
	}
	Points2DSingleLocation pointsIn(pointsInData, pointsInDims);

	std::uint32_t pathPointsInDims[2U] = {1193U, 2U};

	// Inject path_points
	auto pathPointsInData = new Single[sizeof(pathPointsInjectionData) / sizeof(pathPointsInjectionData[0])];
	// transpose.
	for (auto i = int(); i < int(pathPointsInDims[0U]); ++i)
	{
		pathPointsInData[i] = Single(pathPointsInjectionData[2 * i]);
		pathPointsInData[i + int(pathPointsInDims[0U])] = Single(pathPointsInjectionData[2 * i + 1]);
	}
	Points2DSingleLocation pathPointsIn(pathPointsInData, pathPointsInDims);

	Single* fPointsData = new Single[10];

	fPointsData[0] = 226.331383f; fPointsData[1] = 149.901333f; fPointsData[2] = 334.084634f;
	fPointsData[3] = 368.256446f; fPointsData[4] = 385.6785f; fPointsData[5] = 48.6657858f;
	fPointsData[6] = 49.9942557f; fPointsData[7] = 38.4442916f; fPointsData[8] = 26.334433f;
	fPointsData[9] = -50.5476241f;

	Points2DSingleLocation fPoints(fPointsData, pointsInDims);





	using Frame2DCharacteristics = GM::UC::FrenetSerret::Points<float, 2U, 8U>;
	
	std::uint32_t numberOfCharecteristicPoints = sizeof(KintejctionData) / sizeof(KintejctionData[0]);
	auto characteristic2DFramData = new Single[Frame2DCharacteristics::m_numberComponentsInMainAxis * numberOfCharecteristicPoints];

	// May use: MATH_UTILITIES::Transpose2D.
	for (auto charecteristicPointItr = int(); charecteristicPointItr < int(numberOfCharecteristicPoints); ++charecteristicPointItr)
	{
		// position
		auto numberOfcomponentsInProperty = (sizeof(pathPointsInjectionData) / sizeof(pathPointsInjectionData[0])) / numberOfCharecteristicPoints;
		for (auto componentIterator = int(); componentIterator < numberOfcomponentsInProperty; ++componentIterator)
		{
			characteristic2DFramData[charecteristicPointItr + numberOfCharecteristicPoints * componentIterator]
				= Single(pathPointsInjectionData[componentIterator + charecteristicPointItr * numberOfcomponentsInProperty]);
		}
		auto offsetForWrite = numberOfcomponentsInProperty * int(numberOfCharecteristicPoints);
		// T
		numberOfcomponentsInProperty = (sizeof(TintejctionData) / sizeof(TintejctionData[0])) / numberOfCharecteristicPoints;
		for (auto componentIterator = int(); componentIterator < numberOfcomponentsInProperty; ++componentIterator)
		{
			characteristic2DFramData[charecteristicPointItr + numberOfCharecteristicPoints * componentIterator + offsetForWrite]
				= Single(TintejctionData[componentIterator + charecteristicPointItr * numberOfcomponentsInProperty]);
		}
		offsetForWrite += numberOfcomponentsInProperty * int(numberOfCharecteristicPoints);
		// N
		numberOfcomponentsInProperty = (sizeof(NinjectionData) / sizeof(NinjectionData[0])) / numberOfCharecteristicPoints;
		for (auto componentIterator = int(); componentIterator < numberOfcomponentsInProperty; ++componentIterator)
		{
			characteristic2DFramData[charecteristicPointItr + numberOfCharecteristicPoints * componentIterator + offsetForWrite]
				= Single(NinjectionData[componentIterator + charecteristicPointItr * numberOfcomponentsInProperty]);
		}
		offsetForWrite += numberOfcomponentsInProperty * int(numberOfCharecteristicPoints);
		// K
		numberOfcomponentsInProperty = (sizeof(KintejctionData) / sizeof(KintejctionData[0])) / numberOfCharecteristicPoints;
		for (auto componentIterator = int(); componentIterator < numberOfcomponentsInProperty; ++componentIterator)
		{
			characteristic2DFramData[charecteristicPointItr + numberOfCharecteristicPoints * componentIterator + offsetForWrite]
				= Single(KintejctionData[componentIterator + charecteristicPointItr * numberOfcomponentsInProperty]);
		}
		offsetForWrite += numberOfcomponentsInProperty * int(numberOfCharecteristicPoints);
		// K'
		numberOfcomponentsInProperty = (sizeof(KTagintejctionData) / sizeof(KTagintejctionData[0])) / numberOfCharecteristicPoints;
		for (auto componentIterator = int(); componentIterator < numberOfcomponentsInProperty; ++componentIterator)
		{
			characteristic2DFramData[charecteristicPointItr + numberOfCharecteristicPoints * componentIterator + offsetForWrite]
				= Single(KTagintejctionData[componentIterator + charecteristicPointItr * numberOfcomponentsInProperty]);
		}
	}

	std::array<uint32_t, 2U> characteresticDims = { numberOfCharecteristicPoints, 8U };
	Frame2DCharacteristics characteristicsFor2DFrame(characteristic2DFramData, characteresticDims.data());


	FrenetSerret2D<float> frame2D(characteristicsFor2DFrame);


	auto start = std::chrono::high_resolution_clock::now();
	///// Start of f to c to f trajectories '{'
	const auto pointsCartesianFromFrenet = frame2D.toCartesianPoints(fPoints);
	const auto pointsFrenetFromCartesian = frame2D.toFrenetPoints(pointsCartesianFromFrenet);
	///// '}' End of f to c to f trajectories
	auto finish = std::chrono::high_resolution_clock::now();
	std::cout << "high_resolution_clock: toCartesianPoints + toFrenetPoints " << __FILE__ << ' ' << __LINE__ << " took " << std::chrono::duration_cast<micro>(finish - start).count() << std::endl;
	

	// Prepare input cartesian trajectories '{'
	using TrajectoriesCartesian = GM::UC::FrenetSerret::TrajectoriesCartesian;

	auto cTrajectories = new Single[sizeof(CTrajectories) / sizeof(CTrajectories[0U])];
	const int numberOfElementsInTrajectory = (sizeof(CTrajectories) / sizeof(CTrajectories[0U])) / static_cast<int>(TrajectoriesCartesian::NUMBER_OF_COMPONENTS);

	{
		for (auto elementItr = int(); elementItr < numberOfElementsInTrajectory; ++elementItr)
		{
			for (auto componentItr = int(); componentItr < static_cast<int>(TrajectoriesCartesian::NUMBER_OF_COMPONENTS); ++componentItr)
			{
				cTrajectories[elementItr + componentItr * numberOfElementsInTrajectory] =
					static_cast<Single>(CTrajectories[elementItr * static_cast<int>(TrajectoriesCartesian::NUMBER_OF_COMPONENTS) + componentItr]);
			}
		}
	}

	Points2DSingleState posPoints(cTrajectories, statesInDims);
	// '}' Prepare input cartesian trajectories


	start = std::chrono::high_resolution_clock::now();
	///// Start of c to f trajectories '{'
	auto frenetState = frame2D.toFrenetStateVectors(posPoints);
	///// '}' End of c to f trajectories
	finish = std::chrono::high_resolution_clock::now();
	std::cout << "high_resolution_clock: toFrenetStateVectors " << __FILE__ << ' ' << __LINE__ << " took " << std::chrono::duration_cast<micro>(finish - start).count() << std::endl;

	// Prepare input frenet trajectories '{'

	using TrajectoriesFrenet = GM::UC::FrenetSerret::TrajectoriesFrenet;


	auto fTrajectories = new Single[sizeof(FTrajectories) / sizeof(FTrajectories[0U])];
	const int numberOfElementsInFrenetTrajectory = (sizeof(FTrajectories) / sizeof(FTrajectories[0U])) / static_cast<int>(TrajectoriesFrenet::NUMBER_OF_COMPONENTS);

	{
		for (auto elementItr = int(); elementItr < numberOfElementsInFrenetTrajectory; ++elementItr)
		{
			for (auto componentItr = int(); componentItr < static_cast<int>(TrajectoriesFrenet::NUMBER_OF_COMPONENTS); ++componentItr)
			{
				fTrajectories[elementItr + componentItr * numberOfElementsInFrenetTrajectory] =
					static_cast<Single>(FTrajectories[elementItr * static_cast<int>(TrajectoriesFrenet::NUMBER_OF_COMPONENTS) + componentItr]);
			}
		}
	}

	Points2DSingleState posFrenetPoints(fTrajectories, fStatesInDims);
	// '}' Prepare input frenet trajectories


	start = std::chrono::high_resolution_clock::now();
	///// Start of f to c trajectories '{'
	auto cartesianState = frame2D.toCartesianStateVectors(posFrenetPoints);
	///// '}' End of f to c trajectories
	finish = std::chrono::high_resolution_clock::now();
	std::cout << "high_resolution_clock: toCartesianStateVectors " << __FILE__ << ' ' << __LINE__ << " took " << std::chrono::duration_cast<micro>(finish - start).count() << std::endl;

}
