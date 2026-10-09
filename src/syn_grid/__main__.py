USE_LEGACY = False

if __name__ == "__main__":
    if USE_LEGACY:
        from syn_grid.legacy.app_legacy import main
    else:
        from syn_grid.app import main

    main()
