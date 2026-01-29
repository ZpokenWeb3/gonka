package propagation

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"

	"github.com/cosmos/cosmos-sdk/crypto/keys/secp256k1"
)

type BundleHeader struct {
	BundleID     [32]byte
	Participant  string
	PocHeight    int64
	PocBlockHash []byte
	RootHash     []byte
	Count        uint32
	Version      uint32
	CreatedAt    int64
	Signature    []byte
}

func MakeBundleID(participant string, pocHeight int64, rootHash []byte, count uint32, version uint32) [32]byte {
	h := sha256.New()
	h.Write([]byte(participant))
	var buf [8]byte
	binary.BigEndian.PutUint64(buf[:], uint64(pocHeight))
	h.Write(buf[:])
	h.Write(rootHash)
	binary.BigEndian.PutUint32(buf[:4], count)
	h.Write(buf[:4])
	binary.BigEndian.PutUint32(buf[:4], version)
	h.Write(buf[:4])
	return sha256.Sum256(h.Sum(nil))
}

type HeaderSigner interface {
	Sign(msg []byte) ([]byte, error)
}

func SignHeader(h BundleHeader, privKey []byte) ([]byte, error) {
	if len(privKey) != 32 {
		return nil, errors.New("invalid private key length")
	}
	key := &secp256k1.PrivKey{Key: privKey}
	msg := headerSigningBytes(h)
	return key.Sign(msg)
}

func SignHeaderWith(h BundleHeader, signer HeaderSigner) ([]byte, error) {
	return signer.Sign(headerSigningBytes(h))
}

func VerifyHeader(h BundleHeader, hexPubKey string) error {
	if h.Signature == nil {
		return errors.New("signature missing")
	}
	pubKeyBytes, err := hex.DecodeString(hexPubKey)
	if err != nil {
		return err
	}
	pubKey := &secp256k1.PubKey{Key: pubKeyBytes}
	msg := headerSigningBytes(h)
	if !pubKey.VerifySignature(msg, h.Signature) {
		return errors.New("signature verification failed")
	}
	return nil
}

func headerSigningBytes(h BundleHeader) []byte {
	buf := bytes.NewBuffer(nil)
	buf.Write(h.BundleID[:])
	buf.WriteString(h.Participant)
	var tmp [8]byte
	binary.BigEndian.PutUint64(tmp[:], uint64(h.PocHeight))
	buf.Write(tmp[:])
	buf.Write(h.PocBlockHash)
	buf.Write(h.RootHash)
	binary.BigEndian.PutUint32(tmp[:4], h.Count)
	buf.Write(tmp[:4])
	binary.BigEndian.PutUint32(tmp[:4], h.Version)
	buf.Write(tmp[:4])
	binary.BigEndian.PutUint64(tmp[:], uint64(h.CreatedAt))
	buf.Write(tmp[:])
	return buf.Bytes()
}
