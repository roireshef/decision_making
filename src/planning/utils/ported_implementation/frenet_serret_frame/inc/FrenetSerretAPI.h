#ifndef FRENET_SERRET_API_H_
#define FRENET_SERRET_API_H_
// For types
#include <cstdint>
// For smart pointers and std::allocator
#include <memory>
#include <type_traits>


/***************************/
namespace GM
{
	namespace UC
	{
		namespace FrenetSerret
		{
			/*************************************** Auxilary types ************************************************/
			template<typename TdataType, std::uint32_t TnumberDimensions, std::uint32_t TcomponentInMainAxis>
			class Points
			{
			public:
				Points() noexcept(true) : m_data(nullptr)
				{
					for (auto i = decltype(TnumberDimensions)(); i < TnumberDimensions; ++i)
					{
						m_dimensions[i] = std::uint32_t();
					}
				}

				Points(TdataType* data, uint32_t dimensions[TnumberDimensions]) noexcept(true)
					: m_data(data)
				{
					for (auto i = decltype(TnumberDimensions)(); i < TnumberDimensions; ++i)
					{
						m_dimensions[i] = dimensions[i];
					}
				}

				~Points()
				{
					// Assumes allocator follows the convecion of the std::allocator, and delete on nullptr is allowed.
					delete[] m_data;
					m_data = nullptr;
				}

				static const std::uint32_t m_numberComponentsInMainAxis = TcomponentInMainAxis;
				uint32_t   m_dimensions[TnumberDimensions]; // in elements of sizeof(TdataType);
				TdataType *m_data; // implementation detail: custom destructor, assignment,  
								   //noexcept move.
			}; // Used to allow sets of coordinates of points as parameters. Layout for 2D is: x0..xn-1,y0,yn-1

			template<typename TdataType, std::uint32_t TnumberDimensions>
			class Coefficients final
			{
			public:
				Coefficients(TdataType* data, uint32_t dimensions[TnumberDimensions]) noexcept(true)
					: m_data(data), m_dimensions(dimensions)
				{

				}

				uint32_t m_dimensions[TnumberDimensions]; // in elements of sizeof(TdataType);
				TdataType *m_data; // implementation detail: custom destructor, assignment,  
								   //noexcept move.
			}; // Used to allow sets of coordinates of points as parameters. Layout for 2D is: x0..xn-1,y0,yn-1

			template <typename Tdata>
			struct MinMaxStruct
			{
				Tdata m_min;
				Tdata m_max;
			};

			template<typename TdataType, std::uint32_t TnumberOfDimensions>
			struct TaylorResultSt
			{
				TdataType m_a[TnumberOfDimensions];
				TdataType m_t[TnumberOfDimensions];
				TdataType m_n[TnumberOfDimensions];
				TdataType m_k;
				TdataType m_kTag;
			};
			template<typename TdataType, std::uint32_t TnumberOfDimensions>
			using TaylorResult = TaylorResultSt<TdataType, TnumberOfDimensions>;

			template<typename Tdata>
			using MinMaxLimits = MinMaxStruct<Tdata>;

			template<typename Tdata>
			struct ClosestSegmentsSt
			{
				std::vector<Tdata> m_distanceSquare;
				std::vector<int>   m_indexOfSegment;
				std::vector<Tdata> m_clippedProgress;
			};

			template<typename Tdata>
			using ClosestSegments = ClosestSegmentsSt<Tdata>;
			/*************************************** Auxilary enumerations *******************************************/

			enum class SplineOrder
			{
				TRAJECTORY_CURVE_SPLINE_FIT_ORDER,
				UNDEFINED_ORDER
			};

			enum class CoordinateSystem
			{
				CARTESIAN,
				FRENET
			};

			// trajectoriesToFrenet
			enum class TrajectoriesCartesian : std::int32_t
			{
				C_X,
				C_Y,
				C_YAW,
				C_V,
				C_A,
				C_K,
				NUMBER_OF_COMPONENTS
			};

			enum class TrajectoriesFrenet : std::int32_t
			{
				FS_SX,
				FS_SV,
				FS_SA,
				FS_DX,
				FS_DV,
				FS_DA,
				NUMBER_OF_COMPONENTS
			};

			/*************************************** Auxilary values *************************************************/

			static float TRAJECTORY_ARCLEN_RESOLUTION = 0.5000634237895241f;
			// Policy check for meeting predefined set of configuations '{'
			template<std::uint32_t Tdimensions, std::uint32_t TnumberOfComponentInMainAxis>
			struct is_supported_dimension_for_points
			{
				static const bool value = (2U == Tdimensions) && (2U == TnumberOfComponentInMainAxis);
			};

			template<typename TdataType>
			struct is_supported_data_type
			{
				static const bool value = std::is_floating_point<TdataType>::value;
			};

			template<typename TdataType, std::uint32_t Tdimensions, std::uint32_t TnumberOfComponentsInMainAxis>
			struct is_supported_frame_points_configuration
			{
				static const bool value = is_supported_data_type<TdataType>::value && is_supported_dimension_for_points<Tdimensions, TnumberOfComponentsInMainAxis>::value;
			};

			template<std::uint32_t Tdimensions, std::uint32_t TnumberOfComponentInMainAxis>
			struct is_supported_dimension_for_state
			{
				static const bool value = (6U == TnumberOfComponentInMainAxis) && (2U == Tdimensions);
			};

			template<typename TdataType, std::uint32_t Tdimensions, std::uint32_t TnumberOfComponentsInMainAxis>
			struct is_supported_frame_state_configuration
			{
				static const bool value = is_supported_data_type<TdataType>::value && is_supported_dimension_for_state<Tdimensions, TnumberOfComponentsInMainAxis>::value;
			};

			namespace MATH_UTILITIES
			{
				template<typename TdataType>
				using Points2DLocation = GM::UC::FrenetSerret::Points<TdataType, 2U, 2U>; // Note that other configurations may not be supported (compile time indication.).

				template<typename TdataType>
				using Taylor2DResult = GM::UC::FrenetSerret::TaylorResult<TdataType, 2U>;

				template<typename TdataType>
				struct ProjectedCartesianPointSt
				{
					ProjectedCartesianPointSt(const Taylor2DResult<TdataType>& taylor2DResult, const TdataType s) noexcept
						: m_taylor2DResult(taylor2DResult), m_s(s)
					{
					}

					ProjectedCartesianPointSt(const ProjectedCartesianPointSt&) = default;

					Taylor2DResult<TdataType>	m_taylor2DResult;
					TdataType					m_s;
				};

				template<typename TdataType>
				using ProjectedCartesianPoint = ProjectedCartesianPointSt<TdataType>;


				template<typename TdataType>
				Taylor2DResult<TdataType> taylorInterp2D(const TdataType& inSValue, TdataType m_ds, const Points2DLocation<TdataType>& pathPoints, const TdataType* k, const TdataType* kTag, const TdataType* t, const TdataType* N);

				template<typename TdataType, bool Tprofile = false>
				ClosestSegments<TdataType> projectOnPiecewiseLinearCurve(const Points2DLocation<TdataType>& points, const Points2DLocation<TdataType>& pathPoints);
				// Note: std::vector is exposed here. This should not be directly linked without compilation into an external compilation unit.
				template<typename TdataType>
				std::vector<ProjectedCartesianPoint<TdataType>> projectCartesianPoints(const Points<TdataType, 2U, 2U>& points, const Points<TdataType, 2U, 2U>& pathPoints, const TdataType* k, const TdataType* kTag, const TdataType* t, const TdataType* m_N, TdataType ds);


				template<typename TdataType>
				TdataType* Transpose2D(TdataType* baseDestination, TdataType* baseSource, int width, int pitchSource, int height)
				{
					if ((nullptr == baseDestination) || (nullptr == baseSource))
					{
						return nullptr;
					}

					for (auto y = int(); y < height; ++y)
					{
						for (auto x = int(); x < width; ++x)
						{
							baseDestination[y * width + x] = baseSource[y * pitchSource + x];
						}
					}
					return baseDestination;
				}
			}
		}
	}
}

/***************************/
namespace GM
{
	namespace UC
	{
		namespace FrenetSerret
		{
			///*************************************** Auxilary types ************************************************/

			///*************************************** Auxilary enumerations *******************************************/

			///*************************************** Auxilary values *************************************************/

			/*************************************** Frame class *************************************************/

			// Note: the code is implemented to meet requirements of 2D cartesian & frenet points.
			template<typename TdataType, std::uint32_t Tdimensions, class Allocator = std::allocator<TdataType>	>
			class Frame
			{
				public:

				template<std::uint32_t TnumberComponentsInMainAxis>
				using Points = GM::UC::FrenetSerret::Points<TdataType, Tdimensions, TnumberComponentsInMainAxis>;
				Frame(const Points<8U>& curveSamples, TdataType ds = TRAJECTORY_ARCLEN_RESOLUTION) noexcept(false) : m_ds(ds), m_isValid(false)
				{
					const auto numberOfElements = curveSamples.m_dimensions[0];
					TdataType* ptrCaracteristicsData = new TdataType[numberOfElements * curveSamples.m_dimensions[1]];
					m_pathPointsIn.m_data = ptrCaracteristicsData;
					m_pathPointsIn.m_dimensions[0U] = numberOfElements;
					m_pathPointsIn.m_dimensions[1U] = 2U;
					
					auto componentOffset = 0;
					auto numberOfDestinationComponents = int(m_pathPointsIn.m_dimensions[1U]);
					// fill position of points.
					/*MATH_UTILITIES::Transpose2D(
						m_pathPointsIn.m_data, 
						curveSamples.m_data + componentOffset,
						numberOfDestinationComponents, 
						curveSamples.m_numberComponentsInMainAxis, 
						numberOfElements
					);*/
					memcpy(m_pathPointsIn.m_data, curveSamples.m_data + componentOffset, curveSamples.m_numberComponentsInMainAxis * numberOfElements * sizeof(TdataType));

					// fill T of points.
					componentOffset += numberOfDestinationComponents;
					m_t = m_pathPointsIn.m_data + numberOfElements * componentOffset;
				/*	MATH_UTILITIES::Transpose2D(
						m_t,
						curveSamples.m_data + componentOffset,
						numberOfDestinationComponents,
						curveSamples.m_numberComponentsInMainAxis,
						numberOfElements
					);*/
					// fill N of points.
					componentOffset += numberOfDestinationComponents;
					m_n = m_pathPointsIn.m_data + numberOfElements * componentOffset;
					/*MATH_UTILITIES::Transpose2D(
						m_n,
						curveSamples.m_data + componentOffset,
						numberOfDestinationComponents,
						curveSamples.m_numberComponentsInMainAxis,
						numberOfElements
					);*/
					// fill K of points.
					componentOffset += numberOfDestinationComponents;
					m_k = m_pathPointsIn.m_data + numberOfElements * componentOffset;
					numberOfDestinationComponents = 1;
			/*		MATH_UTILITIES::Transpose2D(
						m_k,
						curveSamples.m_data + componentOffset,
						numberOfDestinationComponents,
						curveSamples.m_numberComponentsInMainAxis,
						numberOfElements
					);*/
					// fill K' of points.
					componentOffset += numberOfDestinationComponents;
					m_kTag = m_pathPointsIn.m_data + numberOfElements * componentOffset;
			/*		MATH_UTILITIES::Transpose2D(
						m_kTag,
						curveSamples.m_data + componentOffset,
						numberOfDestinationComponents,
						curveSamples.m_numberComponentsInMainAxis,
						numberOfElements
					);*/
					/*
					for (auto elementIterator = int(); elementIterator < numberOfElements; ++elementIterator)
					{
						for (auto componentIterator = int(); componentIterator < numberOfDestinationComponents; ++componentIterator)
						{
							m_pathPointsIn.m_data[elementIterator * m_pathPointsIn.m_dimensions[1U]]
								= curveSamples[elementIterator * curveSamples.m_numberComponentsInMainAxis + (componentIterator + componentOffset)];
						}
					}
					*/

				}

				// Factories, builders and other generating patterns may be placed here.
				static std::unique_ptr<Frame> fit
				(
					Points<2U>&&			cartesianPoints,
					TdataType				ds				= TRAJECTORY_ARCLEN_RESOLUTION,
					SplineOrder				splineOrder		= SplineOrder::TRAJECTORY_CURVE_SPLINE_FIT_ORDER
				) noexcept(false);

				/*************************************** Conversions methods *****************************************************/

				// Conversions of sets of points.
				template<typename Ttype = TdataType, std::uint32_t Dim = Tdimensions, std::uint32_t TnumberComponentsPerElement = TnumberOfElementsInMainAxis,
					typename std::enable_if<is_supported_frame_points_configuration<Ttype, Dim, TnumberComponentsPerElement>::value, int >::type = 0
				>
				Points<TnumberComponentsPerElement> toCartesianPoints
				(
					const Points<TnumberComponentsPerElement>& fromPoints,
					CoordinateSystem		  fromCoordinateSystem = CoordinateSystem::FRENET
				) noexcept(false)
				{
					//using taylorInterp = GM::UC::FrenetSerret::MATH_UTILITIES::taylorInterp2D<TdataType>;

					TdataType* cartesianPointsData = new TdataType[fromPoints.m_dimensions[0U] * 2U];

					for (auto pointIterator = int(); pointIterator < int(fromPoints.m_dimensions[0U]); ++pointIterator)
					{
						auto taylor = GM::UC::FrenetSerret::MATH_UTILITIES::taylorInterp2D<TdataType>
											(fromPoints.m_data[pointIterator], m_ds, m_pathPointsIn, m_k, m_kTag, m_t, m_n);

						for (auto i = int(); i < int(TnumberComponentsPerElement); ++i)
						{
							const auto cartesianComponent = taylor.m_a[i] + taylor.m_n[i] * fromPoints.m_data[pointIterator + fromPoints.m_dimensions[0U]];
							cartesianPointsData[i * fromPoints.m_dimensions[0U] + pointIterator] = cartesianComponent;
						}

						//const auto sVal = tayloredPoints[pointIterator].m_a
					}



					TdataType* ptr = nullptr;
					std::uint32_t dims[Tdimensions] = {fromPoints.m_dimensions[0U], 2U};
					return Points<TnumberComponentsPerElement>(cartesianPointsData, dims);
				}

				template<typename Ttype = TdataType, std::uint32_t Dim = Tdimensions, std::uint32_t TnumberComponentsPerElement = TnumberOfElementsInMainAxis,
					typename std::enable_if<is_supported_frame_points_configuration<Ttype, Dim, TnumberComponentsPerElement>::value, int >::type = 0
				>
				Points<TnumberComponentsPerElement> toFrenetPoints
				(
					const Points<TnumberComponentsPerElement>& fromPoints,
					CoordinateSystem		  fromCoordinateSystem = CoordinateSystem::CARTESIAN
				) noexcept(false)
				{

					auto projectedPoints = GM::UC::FrenetSerret::MATH_UTILITIES::projectCartesianPoints<TdataType>(fromPoints, m_pathPointsIn, m_k, m_kTag, m_t, m_n, m_ds);
					const auto numberOfPoints= int(fromPoints.m_dimensions[0U]);

					auto frenetPointsData = new TdataType[2U * numberOfPoints];

					for (auto pointIterator = int(); pointIterator < numberOfPoints; ++pointIterator)
					{
						auto dPointVal = Single();
						for (auto i = int(); i < int(fromPoints.m_dimensions[1U]); ++i)
						{
							dPointVal += (fromPoints.m_data[pointIterator + fromPoints.m_dimensions[0U] * i] - projectedPoints[pointIterator].m_taylor2DResult.m_a[i]) * projectedPoints[pointIterator].m_taylor2DResult.m_n[i];
						}

						frenetPointsData[pointIterator] = projectedPoints[pointIterator].m_s;
						frenetPointsData[pointIterator + numberOfPoints] = dPointVal;

						//frenetPoints.emplace_back(projectedPoints[pointIterator].m_s,dPointVal);
					}


					std::uint32_t dims[Tdimensions] = {std::uint32_t(numberOfPoints), 2U};
					return Points<TnumberComponentsPerElement>(frenetPointsData, dims);
				}

				template<typename Ttype = TdataType, std::uint32_t Dim = Tdimensions, std::uint32_t TnumberComponentsPerElement = TnumberOfElementsInMainAxis,
					typename std::enable_if<is_supported_frame_state_configuration<Ttype, Dim, TnumberComponentsPerElement>::value, int >::type = 0
				>
				Points<TnumberComponentsPerElement> toCartesianStateVectors(
					const				Points<TnumberComponentsPerElement>& posPoints,
					CoordinateSystem	fromCoordinateSystem = CoordinateSystem::FRENET
				) noexcept(false)
				{
					std::uint32_t dims2Dpoints[2] = { posPoints.m_dimensions[0], posPoints.m_dimensions[1] };
					Points2DSingleLocation pointsInFrenret(posPoints.m_data, dims2Dpoints);
					auto numberOfElementsInTrajectory = int(posPoints.m_dimensions[0]);
					auto cartesianStateData = new TdataType[TnumberComponentsPerElement * numberOfElementsInTrajectory];

					auto fTrajectories = posPoints.m_data;

					/// Start of f to c trajectories

					for (auto pointIterator = int(); pointIterator < numberOfElementsInTrajectory; ++pointIterator)
					{
						const auto& s = posPoints.m_data[pointIterator];
						const auto& sv = posPoints.m_data[pointIterator + static_cast<int>(TrajectoriesFrenet::FS_SV) * posPoints.m_dimensions[0U]];
						const auto& dx = posPoints.m_data[pointIterator + static_cast<int>(TrajectoriesFrenet::FS_DX) * posPoints.m_dimensions[0U]];
						const auto& dv = posPoints.m_data[pointIterator + static_cast<int>(TrajectoriesFrenet::FS_DV) * posPoints.m_dimensions[0U]];
						const auto& da = posPoints.m_data[pointIterator + static_cast<int>(TrajectoriesFrenet::FS_DA) * posPoints.m_dimensions[0U]];
						const auto& sa = posPoints.m_data[pointIterator + static_cast<int>(TrajectoriesFrenet::FS_SA) * posPoints.m_dimensions[0U]];

						// for each point.
						auto taylor = GM::UC::FrenetSerret::MATH_UTILITIES::taylorInterp2D(s, m_ds, m_pathPointsIn, m_k, m_kTag, m_t, m_n);

						const auto thetaR = atan2(taylor.m_t[static_cast<int>(TrajectoriesCartesian::C_Y)], taylor.m_t[static_cast<int>(TrajectoriesCartesian::C_X)]);
						const auto radiusRatio = Single(1.0f) - taylor.m_k * dx;

						const auto dTag = Single() == sv ? Single() : dv / sv;
						const auto dTag2 = Single() == sv ? Single() : (da - dTag * sa) / (sv * sv);

						const auto tanDeltaTheta = dTag / radiusRatio;

						const auto deltaTheta = atan2(dTag, radiusRatio);
						const auto cosDeltaTheta = 1.0f / sqrtf(1.0f + tanDeltaTheta * tanDeltaTheta);
						//const auto sinDeltaTheta = tanDeltaTheta * cosDeltaTheta;

						const auto v = (sv * radiusRatio) / cosDeltaTheta;

						const auto kdDiffrencial = (taylor.m_kTag * dx + taylor.m_k * dTag);
						// compute curvature
						const auto k = ((dTag2 + kdDiffrencial * tanDeltaTheta)
							* (cosDeltaTheta * cosDeltaTheta) / radiusRatio + taylor.m_k)
							* cosDeltaTheta / radiusRatio;
						// derivative of delta_theta(via chain rule : d(sx)->d(t)->d(s))
						const auto deltaThetaTag = radiusRatio / cosDeltaTheta * k - taylor.m_k;

						const auto a = sv * sv / cosDeltaTheta *
							(
								radiusRatio * tanDeltaTheta * deltaThetaTag
								-
								kdDiffrencial
								) + sa * radiusRatio / cosDeltaTheta;
						// compute position(cartesian)
						Single posX[2U];
						for (auto i = int(); i < 2; ++i)
						{
							posX[i] = taylor.m_a[i] + taylor.m_n[i] * dx;
						}

						// compute thetaX
						const auto thetaX = thetaR + deltaTheta;

						// Write back: {posX[0], posx[1], theta, v, a, k, thetaX
						cartesianStateData[TnumberComponentsPerElement * pointIterator]     = posX[0U];
						cartesianStateData[TnumberComponentsPerElement * pointIterator + 1] = posX[1U];
						cartesianStateData[TnumberComponentsPerElement * pointIterator + 2] = thetaX;
						cartesianStateData[TnumberComponentsPerElement * pointIterator + 3] = v;
						cartesianStateData[TnumberComponentsPerElement * pointIterator + 4] = a;
						cartesianStateData[TnumberComponentsPerElement * pointIterator + 5] = k;

					}

					pointsInFrenret.m_data = nullptr;

					return Points<TnumberComponentsPerElement>(cartesianStateData, dims2Dpoints);
				}
				
				template<typename Ttype = TdataType, std::uint32_t Dim = Tdimensions, std::uint32_t TnumberComponentsPerElement = TnumberOfElementsInMainAxis,
					typename std::enable_if<is_supported_frame_state_configuration<Ttype, Dim, TnumberComponentsPerElement>::value, int >::type = 0
				>
				Points<TnumberComponentsPerElement> toFrenetStateVectors(
						const				Points<TnumberComponentsPerElement>& posPoints,
						CoordinateSystem	fromCoordinateSystem = CoordinateSystem::CARTESIAN
				) noexcept(false)
				{
					std::uint32_t dims2Dpoints[2] = {posPoints.m_dimensions[0], posPoints.m_dimensions[1]};
					Points2DSingleLocation pointsInFrenret(posPoints.m_data, dims2Dpoints);
					auto numberOfElementsInTrajectory = int(posPoints.m_dimensions[0]);
					auto frenetStateData = new TdataType[TnumberComponentsPerElement * numberOfElementsInTrajectory];
					
					auto cTrajectories = posPoints.m_data;


					const auto prjct = GM::UC::FrenetSerret::MATH_UTILITIES::projectCartesianPoints(pointsInFrenret, m_pathPointsIn, m_k, m_kTag, m_t, m_n, m_ds);

					auto pointIterator = int();
					for (auto& elementItr : prjct)
					{

						auto distancePointVal = Single();
						for (auto i = int(); i < pointsInFrenret.m_numberComponentsInMainAxis; ++i)
						{
							distancePointVal += (pointsInFrenret.m_data[pointIterator + pointsInFrenret.m_dimensions[0U] * i] - elementItr.m_taylor2DResult.m_a[i]) * elementItr.m_taylor2DResult.m_n[i];
						}

						const auto radiusRatio = Single(1.0f) - distancePointVal * elementItr.m_taylor2DResult.m_k;

						const auto thetaR = std::atan2
						(
							elementItr.m_taylor2DResult.m_t[static_cast<int>(TrajectoriesCartesian::C_Y)],
							elementItr.m_taylor2DResult.m_t[static_cast<int>(TrajectoriesCartesian::C_X)]
						);

						const auto deltaTheta = cTrajectories[pointIterator + static_cast<int>(TrajectoriesCartesian::C_YAW) * numberOfElementsInTrajectory] - thetaR;

						// hint the compiler to use sincos / sincosf
						const auto cosDeltaTheta = std::cos(deltaTheta);
						const auto sinDeltaTheta = std::sin(deltaTheta);
						const auto tanDeltaTheta = sinDeltaTheta / cosDeltaTheta;


						const auto sV = cTrajectories[pointIterator + static_cast<int>(TrajectoriesCartesian::C_V) * numberOfElementsInTrajectory] * cosDeltaTheta / radiusRatio;
						const auto dV = cTrajectories[pointIterator + static_cast<int>(TrajectoriesCartesian::C_V) * numberOfElementsInTrajectory] * sinDeltaTheta;

						// derivative of delta_theta(via chain rule : d(sx)->d(t)->d(s))
						const auto deltaThetaTag = radiusRatio / cosDeltaTheta * cTrajectories[pointIterator + static_cast<int>(TrajectoriesCartesian::C_K) * numberOfElementsInTrajectory] - elementItr.m_taylor2DResult.m_k;



						const auto dTag = radiusRatio * tanDeltaTheta;  // invalid: (radius_ratio * np.sin(delta_theta)) ** 2
						const auto dTag2 = -(elementItr.m_taylor2DResult.m_kTag * distancePointVal + elementItr.m_taylor2DResult.m_k * dTag) * tanDeltaTheta
							+
							(
								radiusRatio / (cosDeltaTheta * cosDeltaTheta)
								*
								(
									cTrajectories[pointIterator + static_cast<int>(TrajectoriesCartesian::C_K) * numberOfElementsInTrajectory]
									*
									radiusRatio / cosDeltaTheta
									-
									elementItr.m_taylor2DResult.m_k
								)
							);

						const auto sA =
							(
								cTrajectories[pointIterator + static_cast<int>(TrajectoriesCartesian::C_A) * numberOfElementsInTrajectory]
								-
								(sV * sV)
								/
								cosDeltaTheta
								*
								(
									radiusRatio * tanDeltaTheta * deltaThetaTag
									-
									(
										elementItr.m_taylor2DResult.m_kTag * distancePointVal + elementItr.m_taylor2DResult.m_k * dTag
										)
									)
								)
							*
							(
								cosDeltaTheta / radiusRatio
								);

						const auto dA = dTag2 * (sV * sV) + dTag * sA;

						frenetStateData[TnumberComponentsPerElement * pointIterator]     = prjct[pointIterator].m_s;
						frenetStateData[TnumberComponentsPerElement * pointIterator + 1] = sV;
						frenetStateData[TnumberComponentsPerElement * pointIterator + 2] = sA;
						frenetStateData[TnumberComponentsPerElement * pointIterator + 3] = distancePointVal;
						frenetStateData[TnumberComponentsPerElement * pointIterator + 4] = dV;
						frenetStateData[TnumberComponentsPerElement * pointIterator + 5] = dA;
						++pointIterator;
					}


					pointsInFrenret.m_data = nullptr;



					return Points<TnumberComponentsPerElement>(frenetStateData, dims2Dpoints);
				}

				/*************************************** Get proporties ************************************************/

				bool isValid() const noexcept(true) 
				{
					return m_isValid;
				}

				TdataType getEffectiveDs() const noexcept(true) // This assumption is actually OK here.
				{
					return m_ds;
				}


				/*************************************** Resource management ************************************************/


				void releaseAuxilaryResources()
				{
					m_cartesianPoints.swap(decltype(m_cartesianPoints)());
				}

			private:
				bool	  m_isValid = true;
				TdataType m_ds = TRAJECTORY_ARCLEN_RESOLUTION;
				MinMaxLimits<TdataType> m_sLimits = {0.0f, 100.0f};

			//	TdataType taylorInterp(const TdataType& inSValue);

				/**************************************** Characteristics for the FrenetFrame ********************************/
				using Points2DSingleLocation = GM::UC::FrenetSerret::Points<TdataType, 2U, 2U>; // Note that other configurations may not be supported (compile time indication.).
				Points2DSingleLocation m_pathPointsIn = Points2DSingleLocation();
				TdataType* m_k = nullptr;
				TdataType* m_kTag = nullptr;
				TdataType* m_t = nullptr;
				TdataType* m_n = nullptr;


				/**************************************** Auxilary data members **********************************************/
				std::vector<TdataType> m_cartesianPoints;

			};

		}
	}
}


template<typename TdataType, std::uint32_t Tdimensions, class Allocator = std::allocator<TdataType> >
std::unique_ptr<GM::UC::FrenetSerret::Frame<TdataType, Tdimensions, Allocator>> GM::UC::FrenetSerret::Frame<TdataType, Tdimensions, Allocator>::fit
(
	Points<2U>&& cartesianPoints, 
	TdataType ds,
	SplineOrder splineOrder
)
{
	// TODO: replace with match input with bank, then read components, than generate an instance.
	return std::make_unique<Frame<TdataType, Tdimensions, Allocator>>();
};

/********************** Utility functions *********/

namespace GM
{
	namespace UC
	{
		namespace FrenetSerret
		{
			namespace MATH_UTILITIES
			{
				template<typename TdataType>
				using Points2DLocation = GM::UC::FrenetSerret::Points<TdataType, 2U, 2U>; // Note that other configurations may not be supported (compile time indication.).

				template<typename TdataType>
				using Taylor2DResult = GM::UC::FrenetSerret::TaylorResult<TdataType, 2U>;

				template<typename TdataType>
				using ProjectedCartesianPoint = ProjectedCartesianPointSt<TdataType>;

				template<typename TdataType>
				Taylor2DResult<TdataType> taylorInterp2D(const TdataType& inSValue, TdataType m_ds, const Points2DLocation<TdataType>& pathPoints, const TdataType* m_k, const TdataType* m_kTag, const TdataType* m_t, const TdataType* m_N)
				{
					//assert(inSValue > m_sLimits.m_min && inSValue < m_sLimits.m_max);
					Taylor2DResult<TdataType> rv;
					const auto progressDs = inSValue / m_ds;
					const auto  OindexFloat = roundf(progressDs);
					const auto deltaS = (progressDs - OindexFloat) * m_ds;

					const auto OindexInt = int(OindexFloat);

					auto m_Os = pathPoints;
					// Fetch components:
					rv.m_k = m_k[OindexInt];
					rv.m_kTag = m_kTag[OindexInt];
					// Calculate for K and the derivatives
					const auto kk = rv.m_k * rv.m_k;
					const auto ks = rv.m_k + deltaS * rv.m_kTag;
					const auto ksTag = rv.m_kTag;


					for (auto dimensionItr = int(); dimensionItr < int(pathPoints.m_dimensions[1U]); ++dimensionItr)
					{
						const auto o = m_Os.m_data[OindexInt + pathPoints.m_dimensions[0U] * dimensionItr];
						const auto t = m_t[OindexInt + pathPoints.m_dimensions[0U] * dimensionItr];
						const auto n = m_N[OindexInt + pathPoints.m_dimensions[0U] * dimensionItr];

						// Note: under some circumstances the order of multiplications may cause deviation in results.
						const auto kn = rv.m_k * n;
						const auto kt = rv.m_k * t;

						// Using Horner scheme here:
						rv.m_a[dimensionItr] = o + deltaS *
							( // delta ^ 1
								t + deltaS *
								( // delta ^ 2
									0.5f * kn + deltaS *
									( // delta ^ 3
									(-1.0f / 6.0f) * kk * t
										)
									)
								);

						rv.m_t[dimensionItr] = t + deltaS *
							( // delta ^ 1
								kn + deltaS *
								( // delta ^ 2
									-0.5f * kk * t
									)
								);

						rv.m_n[dimensionItr] = n + deltaS *
							(	// delta ^ 1
								-kt + deltaS *
								( // delta ^ 2
									-0.5f * kk * n
									)
								);
					}

					rv.m_k += deltaS * rv.m_kTag;

					// Retire the pointer hijacked for the operation.
					m_Os.m_data = nullptr;

					// return....
					return rv;
				}

				template<typename TdataType, bool Tprofile>
				ClosestSegments<TdataType> projectOnPiecewiseLinearCurve(const Points2DLocation<TdataType>& points, const Points2DLocation<TdataType>& pathPoints)
				{
					using PointsMatrixType = std::remove_const<std::remove_reference<decltype(points)>::type>::type;

					std::uint32_t segmentsVecDims[2U] = { pathPoints.m_dimensions[0] - 1U, pathPoints.m_dimensions[1] };
					auto segmentsVecData = new TdataType[segmentsVecDims[0U] * segmentsVecDims[1U]];
					auto segmentsVecLenSqr = new TdataType[segmentsVecDims[0U]];

					auto progressMatrixData = new TdataType[(int(pathPoints.m_dimensions[0U]) - 1) * points.m_dimensions[0U]];
					auto progressMatrixClippedData = new TdataType[(int(pathPoints.m_dimensions[0U]) - 1) * points.m_dimensions[0U]];


					ClosestSegments<TdataType> closestSegments;
					closestSegments.m_distanceSquare.reserve(points.m_dimensions[0U]);
					for (auto i = int(); i < int(closestSegments.m_distanceSquare.capacity()); ++i)
					{
						closestSegments.m_distanceSquare.emplace_back(std::numeric_limits<TdataType>::max());
					}
					closestSegments.m_clippedProgress.reserve(points.m_dimensions[0U]);
					for (auto i = int(); i < int(closestSegments.m_clippedProgress.capacity()); ++i)
					{
						closestSegments.m_clippedProgress.emplace_back(TdataType());
					}


					closestSegments.m_indexOfSegment.reserve(points.m_dimensions[0U]);
					for (auto i = int(); i < int(closestSegments.m_indexOfSegment.capacity()); ++i)
					{
						closestSegments.m_indexOfSegment.emplace_back(-1);
					}

					PointsMatrixType segmentsVec(segmentsVecData, segmentsVecDims);
					const int segmentsLength = int(pathPoints.m_dimensions[0U]) - 1;

					// diff + length of diff
					for (auto elementItr = int(); elementItr < int(pathPoints.m_dimensions[1U]); ++elementItr)
					{
						auto offsetToElementBaseAddressRead = elementItr * pathPoints.m_dimensions[0U];
						auto offsetToElementBaseAddressWrite = offsetToElementBaseAddressRead - elementItr;

						auto currentSampleElement = pathPoints.m_data[offsetToElementBaseAddressRead];
						for (auto segmentItr = int(); segmentItr < segmentsLength; ++segmentItr)
						{
							const auto nextSampleElement = pathPoints.m_data[segmentItr + 1U + offsetToElementBaseAddressRead];
							const auto diffVal = nextSampleElement - currentSampleElement;
							segmentsVecData[segmentItr + offsetToElementBaseAddressWrite] = diffVal;
							currentSampleElement = nextSampleElement;

							segmentsVecLenSqr[segmentItr] = (0 == elementItr) ? diffVal * diffVal : (diffVal * diffVal + segmentsVecLenSqr[segmentItr]);
						}
					}

					std::uint32_t segmentsStartDims[2U] = { pathPoints.m_dimensions[0U], pathPoints.m_dimensions[1U] };
					PointsMatrixType segmentsStart(pathPoints.m_data, segmentsStartDims);

					// Create a progress & clipped progress matrices.

					for (auto pointI = int(); pointI < decltype(pointI)(points.m_dimensions[0U]); ++pointI)
					{
						for (auto segmentItr = int(); segmentItr < decltype(segmentItr)(segmentsVec.m_dimensions[0U]); ++segmentItr)
						{
							auto sum = TdataType();
							for (auto segmentComponentIter = int(); segmentComponentIter < decltype(segmentComponentIter)(segmentsVec.m_dimensions[1U]); ++segmentComponentIter)
							{
								sum += (points.m_data[segmentComponentIter * points.m_dimensions[0U] + pointI] - segmentsStart.m_data[segmentComponentIter * segmentsStart.m_dimensions[0U] + segmentItr]) * segmentsVec.m_data[segmentComponentIter * segmentsVec.m_dimensions[0U] + segmentItr];
							}
							const auto progressValue = sum / segmentsVecLenSqr[segmentItr];
							progressMatrixData[pointI + points.m_dimensions[0U] * segmentItr] = progressValue;
							// This supports pre std::clamp versions
							const auto clippedProgressValue = std::min(std::max(progressValue, TdataType(0.0f)), TdataType(1.0f));
							progressMatrixClippedData[pointI + points.m_dimensions[0U] * segmentItr] = clippedProgressValue;

							auto distanceSqrToClippedProjection = TdataType();
							// Fused into this is the creation of clippedProjections 3D matrix.
							for (auto i = decltype(segmentItr)(); i < decltype(segmentItr)(points.m_numberComponentsInMainAxis/*.m_dimensions[1U]*/); ++i)
							{
								auto clippedProjectionsValue =
									clippedProgressValue * segmentsVec.m_data[segmentItr + segmentsLength * i] + segmentsStart.m_data[segmentItr + segmentsStart.m_dimensions[0U] * i];

								const auto distanceOfCurrentComponentPointToSegment = points.m_data[i * points.m_dimensions[0U] + pointI] - clippedProjectionsValue;
								distanceSqrToClippedProjection += distanceOfCurrentComponentPointToSegment * distanceOfCurrentComponentPointToSegment;

							}

							if (closestSegments.m_distanceSquare[pointI] > distanceSqrToClippedProjection)
							{
								closestSegments.m_distanceSquare[pointI] = distanceSqrToClippedProjection;
								closestSegments.m_indexOfSegment[pointI] = segmentItr;
							}

						}
						closestSegments.m_clippedProgress[pointI] = progressMatrixClippedData[pointI + points.m_dimensions[0U] * closestSegments.m_indexOfSegment[pointI]];
					}

					auto isPointInFrontOfCurve = false;
					for (auto pointI = int(); pointI < decltype(pointI)(points.m_dimensions[0U]); ++pointI)
					{
						isPointInFrontOfCurve =
							isPointInFrontOfCurve
							||
							(
							(closestSegments.m_indexOfSegment[pointI] == (segmentsLength - 1))
								&&
								(progressMatrixData[pointI + points.m_dimensions[0U] * (segmentsVec.m_dimensions[0U] - 1)])
								);
						if (isPointInFrontOfCurve)
						{
							std::cout << "Can't project point " << closestSegments.m_indexOfSegment[pointI] << " on curve [" << pathPoints.m_data[0U] << ", " << pathPoints.m_data[pathPoints.m_dimensions[0U]] <<
								"..., " << pathPoints.m_data[pathPoints.m_dimensions[0U] - 1U] << ", " << pathPoints.m_data[2U * pathPoints.m_dimensions[0U] - 1U] << "]\n";
						}
					}

					auto isPointInBackOfCurve = false;
					for (auto pointI = int(); pointI < decltype(pointI)(points.m_dimensions[0U]); ++pointI)
					{
						isPointInBackOfCurve =
							isPointInBackOfCurve
							||
							(
							(closestSegments.m_indexOfSegment[pointI] == int())
								&&
								(progressMatrixData[pointI] < TdataType())
								);
						if (isPointInBackOfCurve)
						{
							std::cout << "Can't project point " << closestSegments.m_indexOfSegment[pointI] << " on curve [" << pathPoints.m_data[0U] << ", " << pathPoints.m_data[pathPoints.m_dimensions[0U]] <<
								"..., " << pathPoints.m_data[pathPoints.m_dimensions[0U] - 1U] << ", " << pathPoints.m_data[2U * pathPoints.m_dimensions[0U] - 1U] << "]\n";
						}
					}
					segmentsStart.m_data = nullptr; // make sure convient class does not delete the data.

					return closestSegments;
				}

				template<typename TdataType>
				std::vector<ProjectedCartesianPoint<TdataType>> projectCartesianPoints(const Points<TdataType, 2U, 2U>& points, const Points<TdataType, 2U, 2U>& pathPoints, const TdataType* m_k, const TdataType* m_kTag, const TdataType* m_t, const TdataType* m_N, TdataType ds)
				{
					static const Single TINY_CURVATURE = 0.0001f;
				
					std::vector<GM::UC::FrenetSerret::MATH_UTILITIES::ProjectedCartesianPoint<TdataType>> rv;
					rv.reserve(points.m_dimensions[0U]);
				
					const auto prjct = projectOnPiecewiseLinearCurve(points, pathPoints);
					for (auto pointItr = 0; pointItr < int(points.m_dimensions[0U]); ++pointItr)
					{
						auto sApprox = (prjct.m_clippedProgress[pointItr] + prjct.m_indexOfSegment[pointItr]) * ds;
						auto taylor = GM::UC::FrenetSerret::MATH_UTILITIES::taylorInterp2D(sApprox, ds, pathPoints, m_k, m_kTag, m_t, m_N);
						const auto isCurvatureBigEnough = fabsf(taylor.m_k) > TINY_CURVATURE;
						if (isCurvatureBigEnough)
						{
							const auto signedRadius = 1.0f / taylor.m_k;
							// vector from the circle center to the input point
							Single centerToPoint[2U]; //////
							auto distanceCenterToPointSqr = Single();
							auto stepNorm = Single();
							auto unNormalizedAngleToPoint = Single();
							for (auto dimension = int(); dimension < int(points.m_numberComponentsInMainAxis/*.m_dimensions[1U]*/); ++dimension)
							{
								const auto pointComponentVal = points.m_data[pointItr + points.m_dimensions[0U] * dimension];
								const auto aComponentVal = taylor.m_a[dimension];
								const auto nComponentVal = taylor.m_n[dimension];
								centerToPoint[dimension] = pointComponentVal - (aComponentVal + nComponentVal * signedRadius);
								distanceCenterToPointSqr += centerToPoint[dimension] * centerToPoint[dimension];
								unNormalizedAngleToPoint += nComponentVal * centerToPoint[dimension];
				
								stepNorm += (pointComponentVal - aComponentVal) * taylor.m_t[dimension];
							}
				
							const auto cosCenterToPoint =
											std::min(fabsf(unNormalizedAngleToPoint / sqrt(distanceCenterToPointSqr)), Single(1.0f));
				
							// arc length from a_s to the new guess point
							const auto stepMagnitude = acos(cosCenterToPoint) * fabsf(signedRadius);
							// get sign from norm.
							const auto step = copysignf(stepMagnitude, stepNorm);
							sApprox += step;
							taylor = GM::UC::FrenetSerret::MATH_UTILITIES::taylorInterp2D(sApprox, ds, pathPoints, m_k, m_kTag, m_t, m_N);
						}
						rv.emplace_back(taylor, sApprox);
					}
					return rv;
				}
			}
		}
	}
}






#endif /* FRENET_SERRET_API_H_ */
