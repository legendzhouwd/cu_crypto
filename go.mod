module github.com/legendzhouwd/cu_crypto

go 1.16

require (
	github.com/consensys/gnark-crypto v0.9.1
	github.com/stretchr/testify v1.8.2
	github.com/xuperchain/crypto v0.0.0-20230728040913-ea9045636ba9
	golang.org/x/crypto v0.10.0
)

// replace github.com/legendzhouwd/cu_crypto => ./

// replace github.com/legendzhouwd/crypto/gm/gmsm/sm3 v0.0.0 => ./core/gmsm/sm3
// replace golang.org/x/crypto v0.10.0 => golang.org/x/crypto v0.14.0
