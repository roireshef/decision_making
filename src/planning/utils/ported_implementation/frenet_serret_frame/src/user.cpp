/* This file is not meant for production.
The purpose of the file is to demonstrate basic usage of the C++ API, and provide some sanity check.
*/

#include <algorithm>
#include <vector>
#include <limits>
#include <cmath>

#include <array>


#include <iostream>
#include <chrono>


#include <complex>
#include <cstdlib>
#include <iostream>
#include <map>
#include <string>

#include "cnpy.h"

// Include the exposed API for the module.
#include "FrenetSerretAPI.h"

#include "injectionData.h"

#include "FakeSegments.h"

// Names aliasing:
using Single = float;
template<typename T>					  
using FrenetSerret2D		 = GM::UC::FrenetSerret::Frame<T, 2U, std::allocator<T>>;
template<std::uint32_t TnumberDimensions, std::uint32_t TnumberComponensInMainAxis> 
using PointsSingle			 = GM::UC::FrenetSerret::Points<float, TnumberDimensions, TnumberComponensInMainAxis>;
using Points2DSingleLocation = GM::UC::FrenetSerret::Points<float, 2U, 2U>; // Note that other configurations may not be supported (compile time indication.).
using Points2DDoubleState    = GM::UC::FrenetSerret::Points<double, 2U, 6U>; // Note that other configurations may not be supported (compile time indication.).

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
bool casePointsCFC(FrenetSerret2D<T>& frame);

template<typename T>
bool casePointFCF(FrenetSerret2D<T>& frame);

template<typename T>
bool caseStateCFC(FrenetSerret2D<T>& frame);

template<typename T>
bool caseStateFCF(FrenetSerret2D<T>& frame);

template<typename T>
bool caseProjectCartesianPointFivePointsProjectionAccurate(FrenetSerret2D<T>& frame);

template<typename T>
bool casefitFrenetOriginalRoutePointsAreProjectedErrorsAreLowEnough(FrenetSerret2D<T>& frame);

int main()
{
	/*********************************** Data injection ********************************************/


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


	/*************************************** Frenet frame instances creation ***************************************************/



	using Frame2DCharacteristics = GM::UC::FrenetSerret::Points<double, 2U, 8U>;
	
	/********************* Example use of hard coded data , in order to construct a Fernet frame 2D object. ******/


	std::uint32_t numberOfCharecteristicPoints = sizeof(KintejctionData) / sizeof(KintejctionData[0]);
	auto characteristic2DFramData = new double[Frame2DCharacteristics::m_numberComponentsInMainAxis * numberOfCharecteristicPoints];

	// May use: MATH_UTILITIES::Transpose2D.
	for (auto charecteristicPointItr = int(); charecteristicPointItr < int(numberOfCharecteristicPoints); ++charecteristicPointItr)
	{
		// position
		auto numberOfcomponentsInProperty = (sizeof(pathPointsInjectionData) / sizeof(pathPointsInjectionData[0])) / numberOfCharecteristicPoints;
		for (auto componentIterator = int(); componentIterator < numberOfcomponentsInProperty; ++componentIterator)
		{
			characteristic2DFramData[charecteristicPointItr + numberOfCharecteristicPoints * componentIterator]
				= double(pathPointsInjectionData[componentIterator + charecteristicPointItr * numberOfcomponentsInProperty]);
		}
		auto offsetForWrite = numberOfcomponentsInProperty * int(numberOfCharecteristicPoints);
		// T
		numberOfcomponentsInProperty = (sizeof(TintejctionData) / sizeof(TintejctionData[0])) / numberOfCharecteristicPoints;
		for (auto componentIterator = int(); componentIterator < numberOfcomponentsInProperty; ++componentIterator)
		{
			characteristic2DFramData[charecteristicPointItr + numberOfCharecteristicPoints * componentIterator + offsetForWrite]
				= double(TintejctionData[componentIterator + charecteristicPointItr * numberOfcomponentsInProperty]);
		}
		offsetForWrite += numberOfcomponentsInProperty * int(numberOfCharecteristicPoints);
		// N
		numberOfcomponentsInProperty = (sizeof(NinjectionData) / sizeof(NinjectionData[0])) / numberOfCharecteristicPoints;
		for (auto componentIterator = int(); componentIterator < numberOfcomponentsInProperty; ++componentIterator)
		{
			characteristic2DFramData[charecteristicPointItr + numberOfCharecteristicPoints * componentIterator + offsetForWrite]
				= double(NinjectionData[componentIterator + charecteristicPointItr * numberOfcomponentsInProperty]);
		}
		offsetForWrite += numberOfcomponentsInProperty * int(numberOfCharecteristicPoints);
		// K
		numberOfcomponentsInProperty = (sizeof(KintejctionData) / sizeof(KintejctionData[0])) / numberOfCharecteristicPoints;
		for (auto componentIterator = int(); componentIterator < numberOfcomponentsInProperty; ++componentIterator)
		{
			characteristic2DFramData[charecteristicPointItr + numberOfCharecteristicPoints * componentIterator + offsetForWrite]
				= double(KintejctionData[componentIterator + charecteristicPointItr * numberOfcomponentsInProperty]);
		}
		offsetForWrite += numberOfcomponentsInProperty * int(numberOfCharecteristicPoints);
		// K'
		numberOfcomponentsInProperty = (sizeof(KTagintejctionData) / sizeof(KTagintejctionData[0])) / numberOfCharecteristicPoints;
		for (auto componentIterator = int(); componentIterator < numberOfcomponentsInProperty; ++componentIterator)
		{
			characteristic2DFramData[charecteristicPointItr + numberOfCharecteristicPoints * componentIterator + offsetForWrite]
				= double(KTagintejctionData[componentIterator + charecteristicPointItr * numberOfcomponentsInProperty]);
		}
	}

	std::array<uint32_t, 2U> characteresticDims = { numberOfCharecteristicPoints, 8U };
	Frame2DCharacteristics characteristicsFor2DFrame(characteristic2DFramData, characteresticDims.data());


	FrenetSerret2D<double> frame2D(characteristicsFor2DFrame);

	/********************* Example use of Numpy serialized data, in order to construct a Fernet frame 2D object. ******/

	// Types that are used to fill the data base of segments.
	SegmentBank<double> segmentBank;
	decltype(segmentBank)::KeyType keyToRetrieve;
	// Add some segments.
	auto addedSegment = segmentBank.addSegment("./data/testSegmentData000");
	keyToRetrieve = addedSegment.getKey();

	/*

	Add more segments, just keep the keys you'd like to use later.

	*/


	// Note: No one prevents you from changing the characteristics of the key, if you really want to do it, you are strongly advised to just create a new segment.

	// use the key.
	const auto segment = segmentBank.findSegment(keyToRetrieve);
	if (!segment.isValid())
	{
		// handle error.
	}
	// Imitate builder pattern:
	const auto ds = segment.getDs();
	const auto ts = segment.getT();
	const auto ns = segment.getN();
	const auto ks = segment.getK();
	const auto kTags = segment.getKTag();
	const auto oPoints = segment.getO();

	const auto numberOfCharacteristicSamples = segment.getNumberOfSamples();
	auto characteristic2DFramDataFromNumpy = new double[Frame2DCharacteristics::m_numberComponentsInMainAxis * numberOfCharecteristicPoints];
	
	
	memcpy(characteristic2DFramDataFromNumpy, oPoints.data(), oPoints.size() * sizeof(double));
	auto offsetInElements = oPoints.size();

	memcpy(characteristic2DFramDataFromNumpy + offsetInElements, ts.data(), ts.size() * sizeof(double));
	offsetInElements += ts.size();

	memcpy(characteristic2DFramDataFromNumpy + offsetInElements, ns.data(), ns.size() * sizeof(double));
	offsetInElements += ns.size();

	memcpy(characteristic2DFramDataFromNumpy + offsetInElements, ks.data(), ks.size() * sizeof(double));
	offsetInElements += ks.size();

	memcpy(characteristic2DFramDataFromNumpy + offsetInElements, kTags.data(), kTags.size() * sizeof(double));
	offsetInElements += kTags.size();


	std::array<uint32_t, 2U> characteresticDimsFromNumpy = { numberOfCharacteristicSamples, 8U };
	Frame2DCharacteristics characteristicsFor2DFrameFromNumpy(characteristic2DFramDataFromNumpy, characteresticDimsFromNumpy.data());
	FrenetSerret2D<double> frame2DFromNumpy(characteristicsFor2DFrameFromNumpy);

	// Just check if both frames are identical:
	std::cout << "\nCMP characteristic data of frames " << memcmp(characteristic2DFramDataFromNumpy, characteristic2DFramData, sizeof(float) * numberOfCharacteristicSamples * 8U) << std::endl;


	/****************************************************************************************************************************/


	auto start = std::chrono::high_resolution_clock::now();
	///// Start of f to c to f trajectories '{'
//	const auto pointsCartesianFromFrenet = frame2DFromNumpy.toCartesianPoints(fPoints);
//	const auto pointsFrenetFromCartesian = frame2DFromNumpy.toFrenetPoints(pointsCartesianFromFrenet);
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

	//Points2DSingleState posPoints(cTrajectories, statesInDims);
	// '}' Prepare input cartesian trajectories


	start = std::chrono::high_resolution_clock::now();
	///// Start of c to f trajectories '{'
//	auto frenetState = frame2DFromNumpy.toFrenetStateVectors(posPoints);
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

	//Points2DSingleState posFrenetPoints(fTrajectories, fStatesInDims);
	// '}' Prepare input frenet trajectories


	start = std::chrono::high_resolution_clock::now();
	///// Start of f to c trajectories '{'
//	auto cartesianState = frame2DFromNumpy.toCartesianStateVectors(posFrenetPoints);
	///// '}' End of f to c trajectories
	finish = std::chrono::high_resolution_clock::now();
	std::cout << "high_resolution_clock: toCartesianStateVectors " << __FILE__ << ' ' << __LINE__ << " took " << std::chrono::duration_cast<micro>(finish - start).count() << std::endl;



	auto testResult = casePointsCFC(frame2DFromNumpy);
	if (!testResult)
	{
		std::cerr << std::endl;
	}
	testResult = casePointFCF(frame2DFromNumpy);
	if (!testResult)
	{
		std::cerr << std::endl;
	}
	testResult = caseStateCFC(frame2DFromNumpy);
	if (!testResult)
	{
		std::cerr << std::endl;
	}
	testResult = caseStateFCF(frame2DFromNumpy);
	if (!testResult)
	{
		std::cerr << std::endl;
	}
	testResult = caseProjectCartesianPointFivePointsProjectionAccurate(frame2DFromNumpy);
	{
		std::cerr << std::endl;
	}

	return 0;
}


template<typename T>
bool casePointsCFC(FrenetSerret2D<T>& frame)
{
	static const auto ACCURACY_TH = 1e-3;  // up to 1[mm] error in euclidean distance
	static const auto kNumberOfControlPoints = 5U;
	static const auto kNumberComponentsPerPoint = 2U;
	std::uint32_t pointsInDims[2U] = {kNumberOfControlPoints, kNumberComponentsPerPoint};

	T* cPointsData = new T[kNumberOfControlPoints * kNumberComponentsPerPoint];
	// Cartesian points data in.
	cPointsData[0] = T(220.0); cPointsData[1] = T(150.0); cPointsData[2] = T(280.0);
	cPointsData[3] = T(320.0); cPointsData[4] = T(370.0); cPointsData[5] = T(0.0);
	cPointsData[6] = T(0.0); cPointsData[7] = T(40.0); cPointsData[8] = T(60.0);
	cPointsData[9] = T(0.0);

	using Points2DLocation = GM::UC::FrenetSerret::Points<T, 2U, 2U>;

	Points2DLocation cPoints(cPointsData, pointsInDims);

	const auto pointsFrenetFromCartesian = frame.toFrenetPoints(cPoints);
	const auto pointsCartesianFromFrenet = frame.toCartesianPoints(pointsFrenetFromCartesian);
	
	auto rv = true;

	for (auto point = 0; point < kNumberOfControlPoints; ++point)
	{
		auto error = double();
		for (auto component = 0; component < kNumberComponentsPerPoint; ++component)
		{
			// sum(norm.L2(pointsActual - pointsExpected))
			auto diff = decltype(error)(cPointsData[point + kNumberOfControlPoints * component] - pointsCartesianFromFrenet.m_data[point + kNumberOfControlPoints * component]);
			error += diff * diff;
		}
		error = std::sqrt(error);
		if (error > ACCURACY_TH)
		{
			std::cerr << "Failed " << __FUNCTION__ << " test point #" << point << " error " << error << "\tFrenetMovingFrame point conversions aren\'t accurate enough" << std::endl;
			rv = false;
		}
	}

	return rv;
}

template<typename T>
bool casePointFCF(FrenetSerret2D<T>& frame)
{
	static const auto ACCURACY_TH = 1e-3;  // up to 1[mm] error in euclidean distance
	static const auto kNumberOfControlPoints = 5U;
	static const auto kNumberComponentsPerPoint = 2U;
	std::uint32_t pointsInDims[2U] = { kNumberOfControlPoints, kNumberComponentsPerPoint };

	T* fPointsData = new T[kNumberOfControlPoints * kNumberComponentsPerPoint];
	// FernetPoints data in.
	fPointsData[0] = T(220.0); fPointsData[1] = T(150.0); fPointsData[2] = T(280.0);
	fPointsData[3] = T(320.0); fPointsData[4] = T(370.0); fPointsData[5] = T(0.0);
	fPointsData[6] = T(0.0); fPointsData[7] = T(40.0); fPointsData[8] = T(60.0);
	fPointsData[9] = T(0.0);

	using Points2DLocation = GM::UC::FrenetSerret::Points<T, 2U, 2U>;

	Points2DLocation fPoints(fPointsData, pointsInDims);

	const auto pointsCartesianFromFrenet = frame.toCartesianPoints(fPoints);
	const auto pointsFrenetFromCartesian = frame.toFrenetPoints(pointsCartesianFromFrenet);

	auto rv = true;

	for (auto point = 0; point < kNumberOfControlPoints; ++point)
	{
		auto error = double();
		for (auto component = 0; component < kNumberComponentsPerPoint; ++component)
		{
			// sum(norm.L2(pointsActual - pointsExpected))
			auto diff = decltype(error)(fPointsData[point + kNumberOfControlPoints * component] - pointsFrenetFromCartesian.m_data[point + kNumberOfControlPoints * component]);
			error += diff * diff;
		}
		error = std::sqrt(error);
		if (error > ACCURACY_TH)
		{
			std::cerr << "Failed " << __FUNCTION__ << " test point #" << point << " error " << error << "\tFrenetMovingFrame point conversions aren\'t accurate enough" << std::endl;
			rv = false;
		}
	}

	return rv;
}


template<typename T>
bool caseStateCFC(FrenetSerret2D<T>& frame)
{
	static const auto POSITION_ACCURACY_TH = 1e-3; // up to 1[mm] error in positions
	static const auto VEL_ACCURACY_TH = 1e-3;  // up to 1[mm / sec] error in velocity
	static const auto ACC_ACCURACY_TH = 1e-3;  // up to 1[mm / sec ^ 2] error in acceleration
	static const auto CURV_ACCURACY_TH = 1e-4; // up to 0.0001[m] error in curvature which accounts to radius of 10, 000[m]

	static const auto kNumberOfControlPoints = 5U;
	static const auto kNumberComponentsPerPoint = 6U;
	std::uint32_t statesInDims[2U] = { kNumberOfControlPoints, kNumberComponentsPerPoint };

	T* cTrajectories = new T[kNumberOfControlPoints * kNumberComponentsPerPoint];
	// Cartesian points data in.
	cTrajectories[0] = T(1.500000000000000000e+02); cTrajectories[1] = T(2.500000000000000000e+02); cTrajectories[2] = T(2.800000000000000000e+02); cTrajectories[3] = T(3.200000000000000000e+02); cTrajectories[4] = T(3.700000000000000000e+02);
	cTrajectories[5] = T(0.000000000000000000e+00); cTrajectories[6] = T(0.000000000000000000e+00); cTrajectories[7] = T(4.000000000000000000e+01); cTrajectories[8] = T(6.000000000000000000e+01); cTrajectories[9] = T(0.000000000000000000e+00);
	cTrajectories[10] = T(-3.926990816987241395e-01); cTrajectories[11] = T(5.235987755982988157e-01); cTrajectories[12] = T(4.487989505128275880e-01); cTrajectories[13] = T(4.487989505128275880e-01); cTrajectories[14] = T(3.490658503988658956e-01);
	cTrajectories[15] = T(1.000000000000000000e+01); cTrajectories[16] = T(2.000000000000000000e+01); cTrajectories[17] = T(1.000000000000000000e+01); cTrajectories[18] = T(3.000000000000000000e+00); cTrajectories[19] = T(2.500000000000000000e+00);
	cTrajectories[20] = T(1.000000000000000000e+00); cTrajectories[21] = T(1.100000000000000089e+00); cTrajectories[22] = T(-9.000000000000000222e-01); cTrajectories[23] = T(-5.000000000000000000e-01); cTrajectories[24] = T(-2.000000000000000000e+00);
	cTrajectories[25] = T(1.000000000000000021e-02); cTrajectories[26] = T(-1.000000000000000021e-02); cTrajectories[27] = T(-5.000000000000000278e-02); cTrajectories[28] = T(-5.000000000000000278e-02); cTrajectories[29] = T(-0.000000000000000000e+00);


	Points2DDoubleState cartesianState(cTrajectories, statesInDims);

	using Points2DLocation = GM::UC::FrenetSerret::Points<T, 2U, 2U>;

	auto frenetState = frame.toFrenetStateVectors(cartesianState);
	auto cartesianStateFromFrenet = frame.toCartesianStateVectors(frenetState);

	auto rv = true;

	static const auto YawX = 0;
	static const auto YawY = 1;
	static const auto Velocity = 3;
	static const auto Acceleration = 4;
	static const auto Curvatura = 5;


	for (auto point = 0; point < kNumberOfControlPoints; ++point)
	{
		auto errorPosition = double();
		// sum(norm.L2(pointsActual - pointsExpected))
		auto diff = decltype(errorPosition)(cTrajectories[point + kNumberOfControlPoints * YawX] - cartesianStateFromFrenet.m_data[point + kNumberOfControlPoints * YawX]);
		errorPosition += diff * diff;
		diff = decltype(errorPosition)(cTrajectories[point + kNumberOfControlPoints * YawY] - cartesianStateFromFrenet.m_data[point + kNumberOfControlPoints * YawY]);
		errorPosition += diff * diff;
		
		errorPosition = std::sqrt(errorPosition);
		if (errorPosition > POSITION_ACCURACY_TH)
		{
			std::cerr << "Failed " << __FUNCTION__ << " test point #" << point << " error " << errorPosition << "\tFrenetMovingFrame position conversions aren\'t accurate enough" << std::endl;
			rv = false;
		}
		
		diff = decltype(errorPosition)(cTrajectories[point + kNumberOfControlPoints * Velocity] - cartesianStateFromFrenet.m_data[point + kNumberOfControlPoints * Velocity]);
		const auto errorVelocity = std::abs(diff);

		if (errorVelocity > VEL_ACCURACY_TH)
		{
			std::cerr << "Failed " << __FUNCTION__ << " test point #" << point << " error " << errorVelocity << "\tFrenetMovingFrame velocity conversions aren\'t accurate enough" << std::endl;
			rv = false;
		}

		diff = decltype(errorPosition)(cTrajectories[point + kNumberOfControlPoints * Acceleration] - cartesianStateFromFrenet.m_data[point + kNumberOfControlPoints * Acceleration]);
		const auto errorAcceleration = std::abs(diff);

		if (errorAcceleration > ACC_ACCURACY_TH)
		{
			std::cerr << "Failed " << __FUNCTION__ << " test point #" << point << " error " << errorAcceleration << "\tFrenetMovingFrame acceleration conversions aren\'t accurate enough" << std::endl;
			rv = false;
		}

		diff = decltype(errorPosition)(cTrajectories[point + kNumberOfControlPoints * Curvatura] - cartesianStateFromFrenet.m_data[point + kNumberOfControlPoints * Curvatura]);
		const auto errorCurvatura = std::abs(diff);

		if (errorCurvatura > CURV_ACCURACY_TH)
		{
			std::cerr << "Failed " << __FUNCTION__ << " test point #" << point << " error " << errorCurvatura << "\tFrenetMovingFrame curvature conversions aren\'t accurate enough" << std::endl;
			rv = false;
		}
	}

	return rv;
}

template<typename T>
bool caseStateFCF(FrenetSerret2D<T>& frame)
{
	static const auto POSITION_ACCURACY_TH = 1e-3; // up to 1[mm] error in positions
	static const auto VEL_ACCURACY_TH = 1e-3;  // up to 1[mm / sec] error in velocity
	static const auto ACC_ACCURACY_TH = 1e-3;  // up to 1[mm / sec ^ 2] error in acceleration

	static const auto kNumberOfControlPoints = 5U;
	static const auto kNumberComponentsPerPoint = 6U;
	std::uint32_t statesInDims[2U] = { kNumberOfControlPoints, kNumberComponentsPerPoint };

	T* fTrajectories = new T[kNumberOfControlPoints * kNumberComponentsPerPoint];
	// Cartesian points data in.
	fTrajectories[0] = T(1.499327059999999960e+02); fTrajectories[1] = T(2.762523039999999810e+02); fTrajectories[2] = T(3.346444640000000277e+02); fTrajectories[3] = T(3.683581540000000132e+02); fTrajectories[4] = T(3.902299810000000093e+02);
	fTrajectories[5] = T(9.104058029999999135e+00); fTrajectories[6] = T(2.883849890000000116e+01); fTrajectories[7] = T(6.857684660000000321e+00); fTrajectories[8] = T(2.469453609999999966e+00); fTrajectories[9] = T(1.270848799999999912e+01);
	 
	fTrajectories[10] = T(1.306959149999999958e+00); fTrajectories[11] = T(5.516704919999999568e-01); fTrajectories[12] = T(-1.843736779999999964e+00); fTrajectories[13] = T(-4.479591019999999979e-01); fTrajectories[14] = T(-7.143794340000000354e+00);
	fTrajectories[15] = T(4.997193759999999685e+01); fTrajectories[16] = T(3.637346000000000146e+01); fTrajectories[17] = T(3.843671179999999765e+01); fTrajectories[18] = T(2.612719769999999997e+01); fTrajectories[19] = T(-5.092887180000000313e+01);
	fTrajectories[20] = T(-3.815291670000000135e+00); fTrajectories[21] = T(-2.560976950000000141e+00); fTrajectories[22] = T(-4.191942369999999585e+00); fTrajectories[23] = T(-4.562550490000000236e-01); fTrajectories[24] = T(9.821187720000000565e-02);
	fTrajectories[25] = T(5.686335169999999772e-01); fTrajectories[26] = T(-9.017563450000000813e+00); fTrajectories[27] = T(-3.637529080000000192e+00); fTrajectories[28] = T(-3.124730580000000257e-01); fTrajectories[29] = T(4.222532949999999730e-01);


	Points2DDoubleState frenetState(fTrajectories, statesInDims);

	using Points2DLocation = GM::UC::FrenetSerret::Points<T, 2U, 2U>;

	auto cartesianState = frame.toCartesianStateVectors(frenetState);
	auto frenetStateFromCartesian = frame.toFrenetStateVectors(cartesianState);

	auto rv = true;

	static const auto PosS = 0;
	static const auto PosD = 3;
	static const auto SV = 1;
	static const auto DV = 4;
	static const auto SA = 2;
	static const auto DA = 5;

	for (auto point = 0; point < kNumberOfControlPoints; ++point)
	{
		auto errorPosition = double();
		// sum(norm.L2(pointsActual - pointsExpected))
		auto diff = decltype(errorPosition)(fTrajectories[point + kNumberOfControlPoints * PosS] - frenetStateFromCartesian.m_data[point + kNumberOfControlPoints * PosS]);
		errorPosition = diff * diff;
		diff = decltype(errorPosition)(fTrajectories[point + kNumberOfControlPoints * PosD] - frenetStateFromCartesian.m_data[point + kNumberOfControlPoints * PosD]);
		errorPosition += diff * diff;

		errorPosition = std::sqrt(errorPosition);
		if (errorPosition > POSITION_ACCURACY_TH)
		{
			std::cerr << "Failed " << __FUNCTION__ << " test point #" << point << " error " << errorPosition << "\tFrenetMovingFrame position conversions aren\'t accurate enough" << std::endl;
			rv = false;
		}

		diff = decltype(errorPosition)(fTrajectories[point + kNumberOfControlPoints * SV] - frenetStateFromCartesian.m_data[point + kNumberOfControlPoints * SV]);
		auto errorVelocity = diff * diff;
		diff = decltype(errorPosition)(fTrajectories[point + kNumberOfControlPoints * DV] - frenetStateFromCartesian.m_data[point + kNumberOfControlPoints * DV]);
		errorVelocity += diff * diff;

		errorVelocity = std::sqrt(errorVelocity);
		if (errorVelocity > VEL_ACCURACY_TH)
		{
			std::cerr << "Failed " << __FUNCTION__ << " test point #" << point << " error " << errorVelocity << "\tFrenetMovingFrame velocity conversions aren\'t accurate enough" << std::endl;
			rv = false;
		}

		diff = decltype(errorPosition)(fTrajectories[point + kNumberOfControlPoints * SA] - frenetStateFromCartesian.m_data[point + kNumberOfControlPoints * SA]);
		auto errorAcceleration = diff * diff;
		diff = decltype(errorPosition)(fTrajectories[point + kNumberOfControlPoints * DA] - frenetStateFromCartesian.m_data[point + kNumberOfControlPoints * DA]);
		errorAcceleration += diff * diff;

		errorAcceleration = std::sqrt(errorAcceleration);
		if (errorAcceleration > ACC_ACCURACY_TH)
		{
			std::cerr << "Failed " << __FUNCTION__ << " test point #" << point << " error " << errorAcceleration << "\tFrenetMovingFrame acceleration conversions aren\'t accurate enough" << std::endl;
			rv = false;
		}

	}

	return rv;
}


template<typename T>
bool caseProjectCartesianPointFivePointsProjectionAccurate(FrenetSerret2D<T>& frame)
{
	static const auto ACCURACY_TH = 1e-3;  // up to 1[mm] error in euclidean distance
	static const auto kNumberOfControlPoints = 5U;
	static const auto kNumberComponentsPerPoint = 2U;
	static const auto FP_SX = 0;
	std::uint32_t pointsInDims[2U] = { kNumberOfControlPoints, kNumberComponentsPerPoint };

	T* fPointsData = new T[kNumberOfControlPoints * kNumberComponentsPerPoint];
	// FernetPoints data in.
	fPointsData[0] = T(220.0); fPointsData[1] = T(150.0); fPointsData[2] = T(280.0);
	fPointsData[3] = T(320.0); fPointsData[4] = T(370.0); fPointsData[5] = T(0.0);
	fPointsData[6] = T(0.0); fPointsData[7] = T(40.0); fPointsData[8] = T(60.0);
	fPointsData[9] = T(0.0);

	using Points2DLocation = GM::UC::FrenetSerret::Points<T, 2U, 2U>;

	Points2DLocation fPoints(fPointsData, pointsInDims);
	
	const auto projectedCpoints = frame.toCartesianPoints(fPoints);
	auto projectedPoints = frame.projectCartesianPoints<2U>(projectedCpoints);


	auto rv = true;

	for (auto point = 0; point < kNumberOfControlPoints; ++point)
	{
		auto error = double();
		auto diff = projectedPoints[point].m_s - fPoints.m_data[FP_SX * kNumberOfControlPoints + point];
		//error = diff;
		if (error > ACCURACY_TH)
		{
			std::cerr << "Failed " << __FUNCTION__ << " test point #" << point << " error " << error << "\tFrenetSerret2DFrame._project_cartesian_points is not accurate" << std::endl;
			rv = false;
		}
	}

	return rv;
}





template<typename T>
bool caseCtrajectoryToFtrajectoryToCtrajectoryZeroVelocityTwoWayConversionAccuratePoseAndVelocity(FrenetSerret2D<T>& frame)
{
	static const auto POSITION_ACCURACY_TH = 1e-3; // up to 1[mm] error in positions
	static const auto VEL_ACCURACY_TH = 1e-3;  // up to 1[mm / sec] error in velocity
	static const auto YAW_ACCURACY_TH = 1e-3; // up to 0.01 [rad] error in yaw

	static const auto kNumberOfControlPoints = 5U;
	static const auto kNumberComponentsPerPoint = 6U;
	std::uint32_t statesInDims[2U] = { kNumberOfControlPoints, kNumberComponentsPerPoint };

	T* cTrajectories = new T[kNumberOfControlPoints * kNumberComponentsPerPoint];
	// Cartesian points data in.
	cTrajectories[0] = T(1.500000000000000000e+02); cTrajectories[1] = T(2.500000000000000000e+02); cTrajectories[2] = T(2.800000000000000000e+02); cTrajectories[3] = T(3.200000000000000000e+02); cTrajectories[4] = T(3.700000000000000000e+02);
	cTrajectories[5] = T(0.000000000000000000e+00); cTrajectories[6] = T(0.000000000000000000e+00); cTrajectories[7] = T(4.000000000000000000e+01); cTrajectories[8] = T(6.000000000000000000e+01); cTrajectories[9] = T(0.000000000000000000e+00);
	cTrajectories[10] = T(-3.926990816987241395e-01); cTrajectories[11] = T(5.235987755982988157e-01); cTrajectories[12] = T(4.487989505128275880e-01); cTrajectories[13] = T(4.487989505128275880e-01); cTrajectories[14] = T(3.490658503988658956e-01);
	cTrajectories[15] = T(1.000000000000000000e+01); cTrajectories[16] = T(2.000000000000000000e+01); cTrajectories[17] = T(1.000000000000000000e+01); cTrajectories[18] = T(3.000000000000000000e+00); cTrajectories[19] = T(2.500000000000000000e+00);
	cTrajectories[20] = T(1.000000000000000000e+00); cTrajectories[21] = T(1.100000000000000089e+00); cTrajectories[22] = T(-9.000000000000000222e-01); cTrajectories[23] = T(-5.000000000000000000e-01); cTrajectories[24] = T(-2.000000000000000000e+00);
	cTrajectories[25] = T(1.000000000000000021e-02); cTrajectories[26] = T(-1.000000000000000021e-02); cTrajectories[27] = T(-5.000000000000000278e-02); cTrajectories[28] = T(-5.000000000000000278e-02); cTrajectories[29] = T(-0.000000000000000000e+00);


	Points2DDoubleState cartesianState(cTrajectories, statesInDims);

	using Points2DLocation = GM::UC::FrenetSerret::Points<T, 2U, 2U>;

	auto frenetState = frame.toFrenetStateVectors(cartesianState);
	auto cartesianStateFromFrenet = frame.toCartesianStateVectors(frenetState);

	auto rv = true;

	static const auto YawX = 0;
	static const auto YawY = 1;
	static const auto Velocity = 3;
	static const auto YawC = 1;

	for (auto point = 0; point < kNumberOfControlPoints; ++point)
	{
		auto errorPosition = double();
		// sum(norm.L2(pointsActual - pointsExpected))
		auto diff = decltype(errorPosition)(cTrajectories[point + kNumberOfControlPoints * YawX] - cartesianStateFromFrenet.m_data[point + kNumberOfControlPoints * YawX]);
		errorPosition += diff * diff;
		diff = decltype(errorPosition)(cTrajectories[point + kNumberOfControlPoints * YawY] - cartesianStateFromFrenet.m_data[point + kNumberOfControlPoints * YawY]);
		errorPosition += diff * diff;

		errorPosition = std::sqrt(errorPosition);
		if (errorPosition > POSITION_ACCURACY_TH)
		{
			std::cerr << "Failed " << __FUNCTION__ << " test point #" << point << " error " << errorPosition << "\tFrenetMovingFrame position conversions aren\'t accurate enough" << std::endl;
			rv = false;
		}

		diff = decltype(errorPosition)(cTrajectories[point + kNumberOfControlPoints * Velocity] - cartesianStateFromFrenet.m_data[point + kNumberOfControlPoints * Velocity]);
		const auto errorVelocity = std::abs(diff);

		if (errorVelocity > VEL_ACCURACY_TH)
		{
			std::cerr << "Failed " << __FUNCTION__ << " test point #" << point << " error " << errorVelocity << "\tFrenetMovingFrame velocity conversions aren\'t accurate enough" << std::endl;
			rv = false;
		}

		if ((1 == point) || (0 == point) || ((kNumberOfControlPoints - 1) == point))
		{
			auto isError = std::abs(cartesianStateFromFrenet.m_data[point + kNumberOfControlPoints * YawX]) > YAW_ACCURACY_TH;
			if (isError)
			{
				std::cerr << "Failed " << __FUNCTION__ << " test point #" << point << " error " << isError << "\tYawY value is too large" << std::endl;
				rv = false;
			}
		}
	}

	return rv;
}