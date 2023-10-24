module github.com/legendzhouwd/cu_crypto

go 1.17

require (
	github.com/consensys/gnark-crypto v0.5.3
	github.com/stretchr/testify v1.8.2
	golang.org/x/crypto v0.10.0
// github.com/legendzhouwd/crypto v0.0.0
// sm2 v0.0.0

)

require (
	github.com/davecgh/go-spew v1.1.1 // indirect
	github.com/pmezard/go-difflib v1.0.0 // indirect
	golang.org/x/sys v0.13.0 // indirect
	gopkg.in/yaml.v3 v3.0.1 // indirect
)

// replace github.com/legendzhouwd/cu_crypto => ./

// replace github.com/legendzhouwd/crypto/gm/gmsm/sm3 v0.0.0 => ./core/gmsm/sm3
// replace golang.org/x/crypto v0.10.0 => golang.org/x/crypto v0.14.0
