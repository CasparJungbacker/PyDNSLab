import numpy as np


def case_setup(
    runmode: int = 0,
    retain: bool = False,
    resume: bool = False,
    retain_operators: bool = False,
    resume_operators: bool = False,
    interpolate_new: bool = False,
    tim: int = 4,
    nsteps: int = 1000,
    fixed_dt: bool = False,
    setdt: float = 4.501e-3,
    co_target: float = 0.5,
    interval: int = 100,
    timing: bool = True,
    statinit: int = 1,
    sbicg: bool = True,
    spcg: bool = False,
    bicgmaxit: int = 1e6,
    bicgtol: float = 1e-3,
    pcgmaxit: int = 300,
    pcgtol: float = 1e-3,
    pcgdiagcomp: float = 3e-1,
    pscheme: int = 2,
    res: int = 32,
    w_scale: float = 0.5,
    l_scale: float = 0.5,
    ctanh: float = 5e-2,
    re_tau: float = 180,
    nu: float = 1.9e-3,
    uf: float = 0.4,
) -> dict:

    if runmode not in [0, 1, 2]:
        raise ValueError("Runmode should be one of [0, 1, 2].")

    if tim not in [1, 2, 3, 4]:
        raise ValueError("Time integration order should be one of [1, 2, 3, 4].")

    if interpolate_new and runmode != 2:
        runmode = 2

    if not isinstance(nsteps, int):
        nsteps = int(nsteps)

    if res % 2 != 0:
        raise ValueError("Resolution must be divisible by 2.")

    # Geometry
    length = 4 * np.pi * l_scale
    width = 2 * np.pi * w_scale
    height = 2

    N1 = round(2 * res * w_scale)
    N2 = round(2 * res * l_scale)
    N3 = res + 2

    # Channel flow Reynolds number
    re_d = re_tau * 17.5

    gx = 2 * (2 * re_tau * nu) ** 2 / (height**3)
    gy = 0
    gz = 0

    dhyd = height / 2
    unom = re_d * nu / dhyd
    uscale = 1.2 * unom
    utnom = np.sqrt(0.5 * height * gx)

    if runmode == 0:
        uf = 0
    elif runmode == 1:
        uf = 0.4
    elif runmode == 2:
        uf = 0

    dt = co_target * (length / N2) / uscale

    case = dict(
        runmode=runmode,
        retain=retain,
        resume=resume,
        retain_operators=retain_operators,
        resume_operators=resume_operators,
        interpolate_new=interpolate_new,
        tim=tim,
        nsteps=nsteps,
        fixed_dt=fixed_dt,
        setdt=setdt,
        co_target=co_target,
        interval=interval,
        timing=timing,
        statinit=statinit,
        sbicg=sbicg,
        spcg=spcg,
        bicgmaxit=bicgmaxit,
        bicgtol=bicgtol,
        pcgmaxit=pcgmaxit,
        pcgtol=pcgtol,
        pcgdiagcomp=pcgdiagcomp,
        pscheme=pscheme,
        res=res,
        w_scale=w_scale,
        l_scale=l_scale,
        engine="scipy",
        N1=N1,
        N2=N2,
        N3=N3,
        ctanh=ctanh,
        length=length,
        width=width,
        height=height,
        re_tau=re_tau,
        nu=nu,
        re_d=re_d,
        gx=gx,
        gy=gy,
        gz=gz,
        dhdy=dhyd,
        u_nom=unom,
        uscale=uscale,
        utnom=utnom,
        u_f=uf,
        dt=dt,
    )

    return case
