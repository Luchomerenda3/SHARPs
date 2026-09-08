from sunpy.net import Fido, attrs as a
import astropy.units as u

DATA_PATH = '/home/merenda/gehme/local_data'

LASCO_PATH = DATA_PATH + '/soho/lasco/level_1/c2/'

AIA171_PATH = DATA_PATH + '/soho/lasco/aia_171/'
AIA193_PATH = DATA_PATH + '/sdo/aia/l1/193/'
AIA304_PATH = DATA_PATH + '/sdo/aia/l1/304/'

SECCHI_PATH = DATA_PATH + '/stereo/secchi/L1'
EUVI195_B_PATH = SECCHI_PATH + '/b/img/euvi/'
EUVI195_A_PATH = SECCHI_PATH + '/a/img/euvi/'
COR2A_PATH = SECCHI_PATH + '/a/img/cor2/'
COR2B_PATH = SECCHI_PATH + '/b/img/cor2/'

if __name__ == "__main__":

    time_range = a.Time("2010-09-20 00:00:00", "2010-09-20 08:00:00")

    # Search data
    print("Searching for LASCO...")
    lasco_c2_search = Fido.search(time_range &
                                  a.Instrument.lasco &
                                  a.Detector.c2 &
                                  a.Physobs.intensity &
                                  a.Sample(10*u.min))
    print("Searching for COR2A...")
    cor2_a_search = Fido.search(time_range &
                                a.Source('STEREO_A') &
                                a.Instrument.secchi &
                                a.Detector.cor2 &
                                a.Physobs.intensity &
                                a.Sample(10 * u.min)
                                )
    print("Searching for COR2B...")
    cor2_b_search = Fido.search(time_range &
                                a.Source('STEREO_B') &
                                a.Instrument.secchi &
                                a.Detector.cor2 &
                                a.Physobs.intensity &
                                a.Sample(10 * u.min)
                                )
    print("Searching for AIA193...")
    aia_193_search = Fido.search(time_range &
                                 a.Instrument.aia &
                                 a.Physobs.intensity &
                                 a.Wavelength(193*u.angstrom) &
                                 a.Sample(10 * u.min)
                                 )
    print("Searching for EUVI195A...")
    euvi_a_195_search = Fido.search(time_range &
                                    a.Source('STEREO_A') &
                                    a.Instrument.secchi &
                                    a.Detector.euvi &
                                    a.Physobs.intensity &
                                    a.Wavelength(195 * u.angstrom) &
                                    a.Sample(10 * u.min)
                                    )
    print("Searching for EUVI195B...")
    euvi_b_195_search = Fido.search(time_range &
                                    a.Source('STEREO_B') &
                                    a.Instrument.secchi &
                                    a.Detector.euvi &
                                    a.Physobs.intensity &
                                    a.Wavelength(195 * u.angstrom) &
                                    a.Sample(10 * u.min)
                                    )

    # Get data
    print("Fetching LASCO...")
    lasco_fetch = Fido.fetch(lasco_c2_search, path=LASCO_PATH+'/2010-09-20/')
    print("Fetching COR2A...")
    cor2_a_fetch = Fido.fetch(cor2_a_search, path=COR2A_PATH+'/2010-09-20/')
    print("Fetching COR2B...")
    cor2_b_fetch = Fido.fetch(cor2_b_search, path=COR2B_PATH+'/2010-09-20/')
    print("Fetching AIA193...")
    aia_193_fetch = Fido.fetch(aia_193_search, path=AIA193_PATH+'/2010-09-20/')
    print("Fetching EUVI195A...")
    euvi_a_195_fetch = Fido.fetch(euvi_a_195_search, path=EUVI195_A_PATH+'/2010-09-20/')
    print("Fetching EUVI195B...")
    euvi_b_195_fetch = Fido.fetch(euvi_a_195_search, path=EUVI195_B_PATH+'/2010-09-20/')



