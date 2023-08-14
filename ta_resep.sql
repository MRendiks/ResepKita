-- phpMyAdmin SQL Dump
-- version 5.2.0
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Waktu pembuatan: 03 Mar 2023 pada 04.35
-- Versi server: 10.4.27-MariaDB
-- Versi PHP: 7.4.33

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `ta_resep`
--

-- --------------------------------------------------------

--
-- Struktur dari tabel `resep`
--

CREATE TABLE `resep` (
  `id_resep` int(11) NOT NULL,
  `nama_resep` varchar(255) NOT NULL,
  `bahan` varchar(5000) NOT NULL,
  `langkah` varchar(5000) NOT NULL,
  `kategori` varchar(100) NOT NULL,
  `gambar` varchar(500) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data untuk tabel `resep`
--

INSERT INTO `resep` (`id_resep`, `nama_resep`, `bahan`, `langkah`, `kategori`, `gambar`) VALUES
(1, 'Ayam Woku Manado', '1 Ekor Ayam Kampung (potong 12)2 Buah Jeruk Nipis2 Sdm Garam3 Ruas Kunyit7 Bawang Merah7 Bawang Putih10 Cabe Merah10 Cabe Rawit Merah (sesuai selera)3 Butir Kemiri2 Batang Sereh2 Lembar Daun Salam2 Ikat Daun KemangiPenyedap Rasa1 1/2 Gelas Air', 'Cuci bersih ayam dan tiriskan. Lalu peras jeruk nipis (kalo gak ada jeruk nipis bisa pake cuka) dan beri garam. Aduk hingga merata dan diamkan selama 5 menit, biar ayam gak bau amis.Goreng ayam tersebut setengah matang, lalu tiriskanHaluskan bumbu menggunakan blender. Bawang merah, bawang putih, cabe merah, cabe rawit, kemiri dan kunyit. Oh iya kasih minyak sedikit yaa biar bisa di blender. Untuk sereh nya di geprek aja terus di buat simpul.Setelah bumbu di haluskan barulah di tumis. Jangan lupa sereh dan daun salamnya juga ikut di tumis. Di tumis sampai berubah warna ya Masukan ayam yang sudah di goreng setengah matang ke dalam bumbu yang sudah di tumis, dan diamkan 5 menit dulu. Biar bumbu meresap. Lalu tuangkan 1 1/2 Gelas air. Lalu tambahkan penyedap rasa (saya 3 Sdt, tapi sesuai selera ya) koreksi rasa dan Biar kan sampai mendidihSetelah masakan mendidih, lalu masukan daun kemangi yang sudah di potong potong. Masak lagi sekitar 10 menit. And taraaaaaaaaaaaaaa  jadi deh Ayam Woku Manadonya.Oh iyaa kalo mau di tambahkan potongan tomat merah juga bisa ko. Sesuai selera aja yaa buibuuuu ', 'ayam', 'ayam woku khas manado.jpg'),
(3, 'Ayam cabai kawin', '1/4 kg ayam3 buah cabai hijau besar7 buah cabai merah rawit3 siung bawang putih2 siung bawang merahsecukupnya Gulasecukupnya Garam1/4 buah tomat merahsecukupnya Airsecukupnya Minyak goreng', 'Panaskan minyak di dalam wajan. Setelah minyak panas masukkan ayam yang sudah dipotong dadu. Goreng hingga matang. Lalu tiriskan.Haluskan bawang putih, bawang merah, cabai hijau dan merah, tomat.Panaskan minyak didalam wajan. Setelah minyak panas, masukkan bumbu yang sudah halus. Tunggu sampai wangi. Masukkan ayam yang sudah di goreng. Tambahkan air, gula dan garam. Tunggu sampai bumbu meresap di ayam. Sajikan.', 'ayam', 'ayam cabai kawin.jpeg'),
(4, 'Ayam Geprek', '250 gr daging ayam (saya pakai fillet)\nSecukupnya gula dan garam\n50\n100 gr tepung ayam serbaguna\nSecukupnya lalapan (kemangi,kol,timun)\nSecukupnya minyak panas\nsambal korek\nSecukupnya cabe rawit merah dan bwg putih\n', 'Goreng ayam seperti ayam krispi\nUlek semua bahan sambal kemudian campur dengan minyak panas bekas goreng ayam\nGeprek ayam kemudian campur dengan sambal,sajikan dengan lalapan \n', 'ayam', ''),
(5, 'Minyak Ayam', '400 gr kulit ayam & lemaknya\n8 siung bawang putih kating, cincang kasar\n1 ruas jahe, geprek\n350 ml minyak goreng\n1 sdm ketumbar bubuk\n', 'Cuci bersih kulit ayam. Sisihkan\nAmbil 50 ml minyak goreng satu bahan yg sudah disiapkan tadi. Tumis jahe hingga harum.\nMasukkan kulit ayam, ketumbar dan bawang. Biarkan hingga keluar minyak alaminya sambil diaduk\naduk.\nTambahkan sisa minyak goreng. Masak hingga kulit ayam mengering. Angkat dan saring.\nSetelah cukup dingin minyak ayam tadi dapat disimpan dalam wadah kering seperti mason jar/botol. Kalo saya, saya masukkan juga sedikit bawang putih goreng.\nKulit ayam yg sudah digoreng kering tadi bisa dinikmati begitu saja sebagai camilan dengan ditaburi sejumput garam lalu aduk rata.\n', 'ayam', ''),
(6, 'Nasi Bakar Ayam', '1 piring nasi\n/ fillet ayam, potong kotak, cuci bersih\nDaun pisang yang sudah dibersihkan\nLidi/tusuk gigi untuk mengapit daun\n1 lbr daun salam\n3 cm serai, memarkan\n1 ikat daun kemangi, opsional\n1 cabe merah besar, iris tipis\n1 sdt kecap manis\nMinyak untuk menumis\n100 ml air\nBumbu Halus:\n1 cabe merah besar\n3 cabe kecil\n2 siung bawang merah\n1 siung bawang putih\n1 kemiri\n1 cm kunyit\n/ ruas jahe\n/ tomat\n1 sdt gula pasir\n1 blok kaldu ayam (aku biasa pake maggy) pake royco, masako\nMerica (dikira2 z takar nya, aku sdkit z sih)\n', 'Tumis bumbu halus, masukkan daun salam, serai\nMasukkan potongan ayam, kecap dan air, masak hingga air menyusut dan koreksi rasa\nMasukkan irisan cabe merah dan daun kemangi, aduk sebentar dan matikan api\nTata daun pisang, tata nasi di atas daun, beri isian ayam atau bisa juga nasi di campur dan diaduk dulu dengan ayam baru di tata di daun Pisang, bunbu ayam akan lebih meresap k nasi\nSetelah nasi di bungkus, barulah nasi di bakar, aku bakar pake teflon z br cepet \nSajikan selagi hangat endeusss\nNote: Jangan membakar terlalu lama biar nasi ga kotor dari daun yg di bakar kering\n', 'ayam', ''),
(7, 'Ayam Saus Hintalu Jaruk', '1/2 Ekor ayam\n2 Butir Hintalu Jaruk\n1 Buah Cabe merah\n1 Buah Cabe hijau\n300 gr Tepung terigu\n1 bks Merica bubuk\n2 Siung Bawang putih\n1 Buah jeruk nipis\nsecukupnya Minyak goreng untuk menumis dan menggoreng\nsecukupnya Gula, garam dan penyedap rasa\n', 'Potong ayam menjadi kotak\nkotak ukuran sedang cuci bersih kemudian lumuri perasan jeruk nipis, garam dan merica\nBuat adonan tepung untuk melapisi ayam. Adonan A (Tepung terigu, garam, merica, penyedap dan air es) dan adonan B (Tepung terigu, garam, merica dan penyedap)\nMasukan daging ayam kedalam adonan A kemudian lapisi dengan adonan B, setelah itu goreng daging kedalam minyak panas sampai berwarna kuning keemasan. Angkat tiriskan\nCincang bawang putih,iris seron cabe merah dan cabe hijau. Lumatkan kuning hintalu jaruk aka telur asin dengan air dingin sampai berbentuk pasta\nTumis bawang putih dan cabe sampai harum, kemudian tuangkan ayam dan pasta kuning telur.\nTambahkan gula, garam, penyedap dan merica secukupnya. Tes rasa dan masal hingga mengental\nAngkat dan sajikan bersama nasi putih panas. Selamat mencoba\n', 'ayam', ''),
(8, 'Ayam saos teriyaki Lada Hitam', 'Ayam bagian dada dan tulang\n1 buah bawang bombay\n2 siung bawang daun\n4 siung bawang putih\n1 buah cabe merah besar\n1 buah cabe ijo besar\nBahan bumbu :\nsecukupnya Air\nMentega untuk menumis\n1 bungkus saos teriyaki (2 sdm)\n1 sdm gula pasir\n1 bungkus royco ayam\nLarutan meizena\n', 'Cara Buat :\n1. Sediakan teplon or wajan beri sedikit minyak dan masukan bawang putih dan bawang bombay yang sudah di potong potong tumis sampai harum. Bisa juga sebelum dimasak ayamnya dibalurkan ke saos teriyaki dan diamkan selama 15 menit agar meresap\n2. Masukan ayam oseng sampai agak mateng, baru masukan cabe ijo dan merah.\nBeri sedikit air\n3. Masukan bumbu royco ayan saos teriyaki, gula, dan bubuk lada hitam. Cicipin\n4. Setelah mulai mateng masukan daun bawang jika ingin kental bisa masukan larutan meizena yak. Saya tadi gak pake karena tiba tiba amnesia kirain punya tepungnya ternyata udah habis  \nSetelah masak siap disajikan dengan nasi angat.\n', 'ayam', ''),
(9, 'Steak ayam', '300 gr dada ayam fillet\n1 sdm air jeruk nipis\nsecukupnya garam\nadonan basah\n50 gr terigu\n10 gr maizena\n1/4 sdt merica\ngaram\nsecukupnya air\nadonan kering\n150 gr terigu\n30 gr maizena\n1/2 sdt merica bubuk\n1/4 sdt kaldu jamur\nsecukupnya garam\nsecukupnya minyak goreng\nbahan saus jamur\n50 gr jamur tiram blender\n3 bawang putih cincang halus\n20 gr keju parut\n1 sdm maizena cairkan dgn air\n2 sdm saus tiram\n1 sdm kecap manis\n1/2 sdt merica halus\nsecukupnya garam & kaldu jamur\nair\nbahan pelengkap\nbuncis dan wortel di kukus\n', 'Cuci bersih ayam, iris tipis melebar, rendam dgn garam dan air jeruk nipis 10 menit, cuci\nCampur adonan kering, sisihkan\nCampur adonan basah, sisihkan\nPanaskan minyak goreng dgn api sedang, gulingkan ayam pada adonan kering, lalu celup ke adonan basah, gulingkan lg pada adonan kering, jgn di remas2, kibas2 daging ayam pada adonan kering, lalu goreng di minyak yg banyak dan panas, biarkan krispinya terbentuk dl, jgn di balik2 dl, setelah cukup kering br di balik, goreng smpe matang, tiriskan\nBlender jamur tiram kasih air sedikit. tumis bawang putih yg di cincang halus, masukkan jamur yg di blender, masukkan bumbu2 yg lain sampai meresap, lalu tuang maizena yg sdh di cairkan dgn setengah gelas air, aduk2, masukkan keju parut, aduk sampai meletup2, tes rasa, klo sdh sesuai, angkat\nTata dlm piring, ayam goreng tepung dan sayur kukus, siram dgn saus jamur, taburi daun bawang atau parsley jk ada.\n', 'ayam', ''),
(10, 'Ayam Saos Asam Manis Simple', '1/4 kg Ayam bagian dada fillet (Potong dadu)\nSecukupnya air jeruk nipis utk baluran\n1 Bungkus tepung bumbu ayam instan uk besar + campur sedikit tepung terigu\n3 Siung b.putih (iris tipis)\n3 Siung b.merah (Iris tipis)\n1 Buah b.bombay (Iris tipis)\n1 Buah tomat uk besar (Haluskan)\n3 sdm saos sambal\n1 Sdm Saos tiram\nSecukupnya Gula & garam\nSecukupnya merica bubuk\n', 'Lumuri ayam yg sdh dipotong dadu dgn garam & perasan jeruk nipis (diamkan sebentar)\nBaluri ayam dengan tepung basah, lanjutkan ke tepung kering, diamkan sebentar\nGoreng ayam yg sdh dbaluri tepung sampai kering & matang, angkat tiriskan.\nMembuat Saos Asam Manis : \nTumis b.merah & putih (sy suka klo bawang y agak kering)\nMasukkan bawang bombay, tumis sebentar.\nMasukkan tomat buah yg sdh dihaluskan tdi, tumis lagi pakai api sedang, tambahkan air secukupnya\nMasukkan saos sambal, saos tiram, sedikit perasan jeruk nipis, gula, garam & merica (koreksi sampai rasanya terasa asam manis gurih segar)\nMasukkan ayam yg sdh digoreng tdi kedalam saos asam manis, diamkan sebentar\nSetelah ayam sdh cukup merasap dengan saos asam manisnya matikan api dan siap disajikan dengan nasi putih hangat\n', 'ayam', ''),
(11, 'Mie Ayam Homemade by suami', 'Bahan mie ayam :\n1 kg tepung terigu proteiin tinggi\n4 butir telur\n3 sendok minyak goreng\nAir\nSecukupnyaa garam\nAir untuk merebus mie :\nsecukupnya Air\nMinyak goreng\nBumbu halus kuah ayam untuk mie ayam :\n3 Dada ayam(potong kotak2 jngn terlalu besar)\n5 siung bawang merah\n5 siung bawang putih\n4 buah kemiri\nsecukupnya Lada\nSecukupnya ketumbar\nJahe\nLengkuas\n2 lembar sereh\n5 lembar daun jeruk\n2 lembar daun salam\nKecap\nGula merah\nMinyak untuk menumis\nBahan pelengkap :\nDaun bawang\nSawi hijau\nBawang goreng(me gak pakai lupa)\nBahan sambal :\n20 buah cabai rawit merah\nGaram(agar tidak cepat bau asam)\nAir untuk merebus\n', 'Kocok telur masukan garam n minyak goreng,aduk rata.masukan tepung terigu uleni pakai tangan sampai kalis(tidak menempel di tangan)diamkan 5 menit.\nBagi adonan menjadi beberapa untuk memudahkan menggilasnya nanti,pipihkn pakai tangan lalu giling pakai ketebalan 4 setalah itu giling lg pakai ketebalan 1(ketebelan selera ya kalau saya suka yg tipis),sebelum di giling ke cetakan mie beri taburan terigu di atas adonannya agar tidak menempel,lakukan sampai adonan habis.\nBagi mie menjadi beberapa porsi jngn lupa di beri taburan terigu ya agar tidak menempel.\nHaluskan semua bumbu halus kecuali daun jeruk&salam.\nSiapkan penggorengan beri minyak untuk menumis bumbu halus,masukan daun jeruk(sobek2)&daun salam,sereh geprek aja,tunggu sampai harum,lalu masukan ayam beri air(banyak boleh karna untuk kuahnya nanti)masukan kecap n gula merah,garam&penyedap test rasa.\nUntuk sambal rebus cabai merah rawit kira2 5menit.\nSiapkan panci beri air untuk merebus mie, masukan minyak goreng ke dalam air tujuaan untuk tidak menyatu.setelah mendidih masukan mie tunggu sampai mie naik ke atas barubdi angkat.\nSiapkan mangkok beri garam n kuah ayam aduk2 mie baru di berikan bahan pelengkap.\nMie ayam homemade by suami siap di sajikan \n', 'ayam', ''),
(12, 'Ayam Bakar Pedas Manis Resep :Nila Sari', '1 Ekor Ayam Broiler Ukuran Jumbo\nBumbu :\n5 Siung Bawang Merah\n10 Buah Cabe (Optional)\n1 Ruas Jahe\n1 Ruas Kunyit\nSecukupnya Lada Bubuk\nSecukupnya Ketumbar\n2 Lembar Daun Jeruk\n3 Lembar Daun Salam\n1 Batang Serai\n4 Siung Bawah Putih\nSecukupnya Kecap\nSecukupnya Gula Merah\nSecukupnya Minyak Goreng\nSecukupnya Royco\nSecukupnya Garam\n1 Bungkus Santan Kara\n', 'Iris tipis bawang merah dan daun jeruk.\nHaluskan cabe, bawang putih, lada, ketumbar, jahe dan kunyit.\nTumis bawang merah hingga wangi.\nMasukan bumbu yang telah dihaluskan ke dalam tumisan bawang tumis semua hingga harum.\nMasukan serai, daun jeruk dan daun salam masak sedikit tuang santan kara dan tambahkan air.\nKemudian masukan ayam taburi royco, lada dan garam aduk aduk tutup dan tunggu ayam empuk dan air mulai susut tes rasa.\nBakar ayam bisa menggunakan arang, teflon, atau batu bakar (saya memakai grill lebih praktis).\nSelamat Mencoba.\n', 'ayam', ''),
(13, 'Semur Ayam Kentang Ala Mama Kirana', '1/2 Ekor ayam\nKentang\nBumbu Halus :\nBawang putih \n bisa di sesuaikan sesuaikan\nBawang merah\nMerica\nKemiri\nSedikit biji Pala\nCabai Merah kriting, bila suka pedas\nSedikit jahe\nRempah:\nDaun salam\nLengkuas\nSereh\nKapulaga\nKulit manis\nIrisan bawang bombay\nIrisan tomat\nGaram\nKecap manis\nSedikit penyedap rasa\nJeruk nipis untuk ayam\n', 'Cuci bersih ayam lalu baluri dengan perasan jeruk nipis\nCuci lagi ayam dan baluri dengan garam lalu rebus sebentar lalu di goreng\nKentang saya goreng bisa di sesuaikan sesuai selera masing\nTips goreng ayam agar hasil nya tdk keras : di goreng hanya sebentar\nTumis bawang bombay lalu masukan bumbu halus dan masukkan semua rempah + irisan tomat + kecap manis\nSetelah bumbu harum masukkan air tunggu gingga mendidih\nTambahkan garam + penyedap rasa apabila kurang manis bisa ditambahkan kecap lagi\nSetelah semua sudah sesuai dgn rasa masukkan kentang goreng + ayam goreng\nTunggu kuah hingga menyusut dan kiraa 5 mnt angkat. Dan terakhir saya taburin bawang goreng bisa disesuaikan jg dgn selera masing. Selesai\n', 'ayam', ''),
(14, 'Ayam suwir', '400 grm dada ayam\n3 lembar daun jeruk\n4 siung bawang putih\n6 siung bawang merah\n1 cm jahe\n1 cm kunyit\nsecukupnya Garam dan gula\nMinyak untuk menumis\n', 'Rebus ayam sampai matang dan di suir suir.\nHaluskan bawang putih, bawang merah, jahe, dan kunyit.\nPanaskan minyak tumis bumbu halus, masukkan daun jeruk masak hingga harum tambahkan air secukupnya.\nMasukkan ayam masak hingga matang.\n', 'ayam', ''),
(15, 'Ayam goreng tepung', '150 gr fillet ayam\n5 sdm tepung bumbu serbaguna\n1 butir telur ayam (kocok lepas)\nGaram sck\nMinya goreng secukupnya untuk menggoreng\n', 'Cuci ayam, tiriskan beri perasan jeruk lemon (skip boleh) Potong ayam bentuk dadu.\nKoco telur beri secukupnya garam. Masukkan ayam yg telah dipotong dadu.\nBalurkan ke dalam tepung serbaguna.\nGoreng pada minyak yg panas.\nSiap disantap\n', 'ayam', ''),
(16, 'Tulang Ayam Pedas Manis', '1/2 kg tulang ayam bersihkan lalu potong\n5 bh cabe merah keriting\n5 bh cabe rawit (tambah jika suka pedas)\n4 butir bawang merah\n2 butir bawang putih\n1 ruas kunyit\n2 lbr daun salam\n1 lbr daun jeruk\n1 btg sereh geprek\n1 ruas jahe geprek\n1 btr jeruk nipis\nsecukupnya kaldu bubuk\nsecukupnya gula pasir\nsecukupnya kecap manis\n500 ml air untuk merebus tulang\nminyak untuk menumis\n', 'Setelah tulang ayam di bersihkan dan di potong potong lumuri tulang dengan perasan jeruk nipis lalu biarkan\nSiapkan bumbu yg akan di haluskan bwg merah, bwg putih, cabe, dan kunyit blender hingga halus\nSiapkan air untuk merebus tulang tambahkan jahe yg sudah di geprek dan daun jeruk ungkep tulang hingga air menyusut\nTumis bumbu yg sudah di halus kan tambahkan air secukupnya dan daun salam juga sereh geprek, masukan tulang ayam, beri kaldu bubuk, gula pasir dan kecap manis secukupnya jgn lupa test rasanya dan biarkan tulang ayam hingga meresap\nSetelah tulang ayam kuahnya agak sedikit meresap angkat lalu sajikan\nBisa untuk teman makan mie ayam lohh moms \n', 'ayam', ''),
(17, 'Ayam penyet rumahan', '1 ekr ayam negri cuci bersih ptng beri air jeruk nipis cuci lagi\n1 buah jeruk nipis\nBumbu ayam goreng jd,yg bsah. bisa beli di pasar,aku beli 3rb\n2 lmbr daun salam\n2 lmbr daun jeruk\n1 btng sere geprek\nGaram secukupnya dan penyedap\nsecukupnya Minyak goreng\nSambel penyet:\n20 buah cabe rawit merah\n5 buah cabe merah\n2 siung bawang p kating\n5 buah bawang merah goreng sebentr\nPenyedap dan sedikit gula\n', 'Ungkep ayam bersama bumbu jadi, daun salam, daun jeruk, sere,dan garam,tambah sedikit penyedap apinya kecil ajah ya. Lalu tutup\nApabila sudah empuk dan matang goreng dalam minyak panas dan banyak sampe warna kuning kecoklatan, angkat\nSementara lagi goreng ayam, kita siapkan sambel dulu, uleg semua cabe, dan bawang merah garam,sedikit gula dan penyedap, setelah agak halus kumpulin sambalnya di tengah cobek lalu siram dengan minyak panas bekas goreng ayam tadi lalu uleg lagi supaya rata\nMasukan ayam goreng ke dalam cobek lalu penyet pake ulegan, siap di hidangkan\nSelamat mencoba\n', 'ayam', ''),
(18, 'Nugget Ayam Home Made', '1/4 kg ayam d fillet ambil dagingnya aja\n1 buah wortel\n1 buah daun bawang prei\n4 siung bawang putih\n8 siung bawang merah\n1 sdt merica\n1/2 kg tepung tapioka\n2 telur ayam\nGaram, penyedap rasa\nTepung roti/ tepung panir\n', 'Cuci bersih ayam. Setelah itu potong kecil2. Masukkan dlm blender. Tambahkan 1 telur ayam, bwg pth, bwg merah, merica. Air sdkt. Lalu blender.\nSetelah itu parut wortel, ptong kecil2 daun bawang prei\nTaruh adonan ayam dalam wadah\nMasukkan wortel, daun prei, tepung tapioka 10 sdm. Kalau pengen lebih padet bisa tambahkan lagi. Sesuai selera\nAduk2 ya sampai tercampur rata. Tambah i garam, penyedap.\nSiapkan wadah. Tuang adonan ke dalam wadah yg sebelumnya d olesi mentega\nKukus d panci pengukusan. Kukus 30 \n 40 mnt.\nSetelah matang, keluarkan dr panci. Biarkan dingin. Lalu keluarkan.\nSiapkan 1 telur (dikocok), tepung panir. Potong2 adonan jd kecil2, celupkan ke telur, lalu baluri dg tepung panir.\nSetelah itu goreng.Nugget siap d hidangkan. Nugget juga bisa d simpan d dalam freezer ya moms . Jadi sewaktu2 bisa langsung goreng.\n', 'ayam', '');

-- --------------------------------------------------------

--
-- Struktur dari tabel `users`
--

CREATE TABLE `users` (
  `id_user` int(11) NOT NULL,
  `username` varchar(100) NOT NULL,
  `password` varchar(50) NOT NULL,
  `last_activity` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data untuk tabel `users`
--

INSERT INTO `users` (`id_user`, `username`, `password`, `last_activity`) VALUES
(1, 'admin', 'admin', '2023-02-28 20:39:45');

--
-- Indexes for dumped tables
--

--
-- Indeks untuk tabel `resep`
--
ALTER TABLE `resep`
  ADD PRIMARY KEY (`id_resep`);

--
-- Indeks untuk tabel `users`
--
ALTER TABLE `users`
  ADD PRIMARY KEY (`id_user`);

--
-- AUTO_INCREMENT untuk tabel yang dibuang
--

--
-- AUTO_INCREMENT untuk tabel `resep`
--
ALTER TABLE `resep`
  MODIFY `id_resep` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=19;

--
-- AUTO_INCREMENT untuk tabel `users`
--
ALTER TABLE `users`
  MODIFY `id_user` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=2;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
