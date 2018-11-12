#ifndef FAKE_SEGMENTS_H_
#define FAKE_SEGMENTS_H_

template<typename TDataType>
class Segment
{
public:

	using KeyType = std::vector<TDataType>;

	bool fillFromDirectory(const std::string& directory) {
		m_isValid = false;
		m_numberOfSamples = decltype(m_numberOfSamples)();
		// Points path
		cnpy::NpyArray os = cnpy::npy_load(directory + "/O.npy");
		const auto oData = os.data<double>();
		// T
		cnpy::NpyArray ts = cnpy::npy_load(directory + "/T.npy");
		const auto tData = ts.data<double>();
		// N
		cnpy::NpyArray ns = cnpy::npy_load(directory + "/N.npy");
		const auto nData = ns.data<double>();
		// K
		cnpy::NpyArray ks = cnpy::npy_load(directory + "/k.npy");
		const auto kData = ks.data<double>();
		// K'
		cnpy::NpyArray kTags = cnpy::npy_load(directory + "/k_tag.npy");
		const auto kTagData = kTags.data<double>();
		// ds
		cnpy::NpyArray ds = cnpy::npy_load(directory + "/ds.npy");
		const auto dsData = ds.data<double>();

		// Points path

		const auto numberOfPoints = int(os.shape[0]);
		m_oContainer.reserve(numberOfPoints * os.shape[1]);
		auto numberOfComponents = int(os.shape[1]);
		fillWithData(m_oContainer, oData, numberOfComponents, numberOfPoints);
		// T
		numberOfComponents = int(ts.shape[1]);
		m_tContainer.reserve(numberOfPoints * numberOfComponents);
		fillWithData(m_tContainer, tData, numberOfComponents, numberOfPoints);
		// N
		numberOfComponents = int(ns.shape[1]);
		m_nContainer.reserve(numberOfPoints * numberOfComponents);
		fillWithData(m_nContainer, nData, numberOfComponents, numberOfPoints);
		// K
		numberOfComponents = int(ks.shape[1]);
		m_kContainer.reserve(numberOfPoints * numberOfComponents);
		fillWithData(m_kContainer, kData, numberOfComponents, numberOfPoints);
		// K'
		numberOfComponents = int(kTags.shape[1]);
		m_kTagContainer.reserve(numberOfPoints * numberOfComponents);
		fillWithData(m_kTagContainer, kTagData, numberOfComponents, numberOfPoints);
		// ds
		m_ds = float(dsData[0]);

		m_isValid = true;
		m_numberOfSamples = numberOfPoints;
		return true;
	}


	const KeyType& getO() const {
		return m_oContainer;
	}

	const KeyType& getKey() const {
		return m_oContainer;
	}

	const std::vector<TDataType>& getT() const {
		return m_tContainer;
	}

	const std::vector<TDataType>& getN() const {
		return m_nContainer;
	}

	const std::vector<TDataType>& getK() const {
		return m_kContainer;
	}

	const std::vector<TDataType>& getKTag() const {
		return m_kTagContainer;
	}

	TDataType getDs() const {
		return m_ds;
	}

	bool isValid() const {
		return m_isValid;
	}

	std::uint32_t getNumberOfSamples() const {
		return m_numberOfSamples;
	}

private:
	std::uint32_t m_numberOfSamples = std::uint32_t();

	KeyType				   m_oContainer;
	std::vector<TDataType> m_tContainer;
	std::vector<TDataType> m_nContainer;
	std::vector<TDataType> m_kContainer;
	std::vector<TDataType> m_kTagContainer;
	TDataType			   m_ds;
	bool				   m_isValid = false;

};

template<typename TDataType>
class SegmentBank
{
public:
	using SegmentType = Segment<TDataType>;
	using KeyType = typename SegmentType::KeyType;

	auto addSegment(const std::string& directory) {

		Segment<TDataType> segment;
		segment.fillFromDirectory(directory);


		m_segments[segment.getKey()] = segment;
		return m_segments[segment.getKey()];
	}

	SegmentType findSegment(const KeyType& oContainer) {
		auto itr = m_segments.find(oContainer);
		if (m_segments.end() == itr)
		{
			return SegmentType();
		}
		return itr->second;
	}
private:
	std::map<KeyType, SegmentType> m_segments;
};



template<typename TdataType, typename TDestDataType>
void fillWithData(std::vector<TDestDataType>& destContainer, const TdataType* srcData, int numberOfComponents, int numberOfPoints)
{
	for (auto component = int(); component < numberOfComponents; ++component)
	{
		for (auto element = int(); element < numberOfPoints; ++element)
		{
			destContainer.push_back(TDestDataType(srcData[numberOfComponents * element + component]));
		}
	}
}

#endif